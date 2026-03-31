// HumanoidPatient.cs (v3 — ROSC sit-up, connected wrist, real compression sink)
// ==============================================================================

using System.Collections;
using UnityEngine;

[RequireComponent(typeof(BoxCollider))]
public class HumanoidPatient : MonoBehaviour
{
    [Header("Appearance")]
    public Color skinColor    = new Color(0.65f, 0.50f, 0.38f);
    public Color shirtColor   = new Color(0.88f, 0.88f, 0.88f);
    public Color trouserColor = new Color(0.32f, 0.32f, 0.38f);

    [Header("Compression Physics")]
    public float maxCompressionDepth = 0.06f;
    public float springStiffness     = 20f;
    public float springDamping       = 6f;

    private HumanoidBuilder.Body _body;
    private float _compressionTarget  = 0f;
    private float _compressionCurrent = 0f;
    private float _compressionVel     = 0f;
    private float _headTiltTarget     = 0f;
    private float _headTiltCurrent    = 0f;
    private bool  _inRecovery         = false;
    private bool  _rosc               = false;
    private float _breathCycle        = 0f;
    private bool  _sittingUp          = false;

    // The WORLD position of the chest — rescuer IK drives toward this
    public Vector3 ChestWorldPosition =>
        _body?.chest != null ? _body.chest.position : transform.position + Vector3.up * 0.22f;

    void Awake()
    {
        // Stand upright in local space; we rotate hips to lay flat
        transform.rotation = Quaternion.identity;
        transform.position = new Vector3(transform.position.x, 0.14f, transform.position.z);

        _body = HumanoidBuilder.Build(transform, skinColor, shirtColor, trouserColor, 1f);

        // Supine: hips rotated so spine chain runs horizontal along world Z
        _body.hips.localEulerAngles = new Vector3(90f, 0f, 0f);
        _body.hips.localPosition    = new Vector3(0f, 0.05f, 0f);

        // Arms rest at sides — rotated to lie along the body
        _body.leftUpperArm.localEulerAngles  = new Vector3(0f, 0f, -25f);
        _body.rightUpperArm.localEulerAngles = new Vector3(0f, 0f,  25f);
        _body.leftForearm.localEulerAngles   = new Vector3(0f, 0f, -10f);
        _body.rightForearm.localEulerAngles  = new Vector3(0f, 0f,  10f);
        _body.leftWrist.localEulerAngles     = Vector3.zero;
        _body.rightWrist.localEulerAngles    = Vector3.zero;

        // Ground collider
        var col = GetComponent<BoxCollider>();
        col.size   = new Vector3(0.5f, 0.25f, 1.8f);
        col.center = new Vector3(0f, 0.12f, 0f);

        BridgeEvents.OnStateUpdate  += HandleState;
        BridgeEvents.OnEpisodeEnd   += HandleEpisodeEnd;
        BridgeEvents.OnPhaseChange  += HandlePhaseChange;
    }

    void OnDestroy()
    {
        BridgeEvents.OnStateUpdate  -= HandleState;
        BridgeEvents.OnEpisodeEnd   -= HandleEpisodeEnd;
        BridgeEvents.OnPhaseChange  -= HandlePhaseChange;
    }

    void Update()
    {
        AnimateChestSpring();
        AnimateHeadTilt();
        AnimateBreathing();
    }

    // ── State handler ─────────────────────────────────────────────────────────
    void HandleState(StatePacket p)
    {
        switch (p.action)
        {
            case 4:  // CHEST COMPRESSIONS — sink proportional to hand placement
                float depth = Mathf.Clamp01(0.4f + (p.vitals?.hand_placement ?? 0.5f) * 0.6f);
                _compressionTarget = maxCompressionDepth * depth;
                break;
            case 2: case 10: // OPEN_AIRWAY / TILT_HEAD_BACK
                _headTiltTarget = 32f;
                break;
            case 5:  // RESCUE BREATHS
                StartCoroutine(ChestRiseTwice());
                break;
            case 8:  // RECOVERY POSITION
                if (!_inRecovery) StartCoroutine(RollToRecovery());
                break;
            default:
                _compressionTarget = 0f;
                break;
        }

        // ROSC — trigger sit-up animation once
        if (p.rosc && !_sittingUp)
        {
            _rosc = true;
            StartCoroutine(SitUpROSC());
        }
    }

    void HandleEpisodeEnd(StatePacket p) => StartCoroutine(ResetPose());
    void HandlePhaseChange(StatePacket p) => StartCoroutine(ResetPose());

    // ── Per-frame animations ──────────────────────────────────────────────────

    void AnimateChestSpring()
    {
        float err = _compressionTarget - _compressionCurrent;
        _compressionVel += (err * springStiffness - _compressionVel * springDamping) * Time.deltaTime;
        _compressionCurrent = Mathf.Clamp(_compressionCurrent + _compressionVel * Time.deltaTime,
                                           0f, maxCompressionDepth);

        if (_body?.chest != null)
        {
            var lp = _body.chest.localPosition;
            _body.chest.localPosition = new Vector3(lp.x, lp.y, -_compressionCurrent);
        }
        // Shoulders follow slightly
        float sf = _compressionCurrent * 0.35f;
        if (_body?.leftShoulder != null)
            _body.leftShoulder.localPosition = new Vector3(_body.leftShoulder.localPosition.x,
                                                            _body.leftShoulder.localPosition.y, -sf);
        if (_body?.rightShoulder != null)
            _body.rightShoulder.localPosition = new Vector3(_body.rightShoulder.localPosition.x,
                                                             _body.rightShoulder.localPosition.y, -sf);
    }

    void AnimateHeadTilt()
    {
        _headTiltCurrent = Mathf.MoveTowards(_headTiltCurrent, _headTiltTarget, 55f * Time.deltaTime);
        if (_body?.neck != null)  _body.neck.localEulerAngles = new Vector3(_headTiltCurrent, 0, 0);
        if (_body?.head != null)  _body.head.localEulerAngles = new Vector3(_headTiltCurrent * 0.45f, 0, 0);
    }

    void AnimateBreathing()
    {
        if (!_rosc) return;
        _breathCycle += Time.deltaTime * 0.35f;
        float rise = Mathf.Max(0f, Mathf.Sin(_breathCycle * Mathf.PI * 2f)) * 0.012f;
        if (_body?.chest != null)
        {
            var lp = _body.chest.localPosition;
            _body.chest.localPosition = new Vector3(lp.x, lp.y + rise, lp.z);
        }
    }

    // ── ROSC sit-up ───────────────────────────────────────────────────────────
    IEnumerator SitUpROSC()
    {
        _sittingUp = true;
        yield return new WaitForSeconds(0.8f);

        // Gradually rotate hips from supine (90°) toward seated (~20°)
        Quaternion supine = _body.hips.localRotation;
        Quaternion seated = Quaternion.Euler(20f, 0f, 0f);
        float t = 0f;
        while (t < 1f)
        {
            t += Time.deltaTime * 0.4f;   // slow, realistic ~2.5s rise
            _body.hips.localRotation = Quaternion.Slerp(supine, seated, Mathf.SmoothStep(0,1,t));
            // Also raise hips off the ground
            _body.hips.localPosition = new Vector3(0f,
                Mathf.Lerp(0.05f, 0.55f, Mathf.SmoothStep(0,1,t)), 0f);
            yield return null;
        }

        // Head lifts, looks around
        t = 0f;
        while (t < 1f)
        {
            t += Time.deltaTime * 1.2f;
            float nod = Mathf.Sin(t * Mathf.PI * 1.5f) * 12f;
            if (_body?.head != null) _body.head.localEulerAngles = new Vector3(nod, nod * 0.3f, 0);
            yield return null;
        }

        // Arms reach forward slightly — natural seated recovery
        if (_body?.leftUpperArm != null)
            _body.leftUpperArm.localEulerAngles = new Vector3(-20f, 0f, -15f);
        if (_body?.rightUpperArm != null)
            _body.rightUpperArm.localEulerAngles = new Vector3(-20f, 0f,  15f);
    }

    // ── Chest rise on rescue breaths ──────────────────────────────────────────
    IEnumerator ChestRiseTwice()
    {
        for (int i = 0; i < 2; i++)
        {
            float t = 0f;
            while (t < 1f)
            {
                t += Time.deltaTime * 1.8f;
                float rise = Mathf.Sin(t * Mathf.PI) * 0.022f;
                if (_body?.chest != null)
                {
                    var lp = _body.chest.localPosition;
                    _body.chest.localPosition = new Vector3(lp.x, lp.y, lp.z - rise);
                }
                yield return null;
            }
            yield return new WaitForSeconds(0.25f);
        }
    }

    // ── Recovery position lateral roll ────────────────────────────────────────
    IEnumerator RollToRecovery()
    {
        _inRecovery = true;
        Quaternion start  = _body.hips.rotation;
        Quaternion target = start * Quaternion.Euler(0f, 0f, 72f);
        Quaternion kneeStart = _body.leftThigh.localRotation;
        Quaternion kneeBent  = Quaternion.Euler(0f, 0f, -55f);
        float t = 0f;
        while (t < 1f)
        {
            t += Time.deltaTime * 0.55f;
            float s = Mathf.SmoothStep(0,1,t);
            _body.hips.rotation = Quaternion.Slerp(start, target, s);
            _body.leftThigh.localRotation = Quaternion.Slerp(kneeStart, kneeBent, s);
            yield return null;
        }
    }

    IEnumerator ResetPose()
    {
        _sittingUp = false; _rosc = false; _inRecovery = false;
        _compressionTarget = 0f; _compressionCurrent = 0f; _compressionVel = 0f;
        _headTiltTarget = 0f; _breathCycle = 0f;

        float t = 0f;
        Quaternion startHips = _body.hips.rotation;
        Quaternion supine    = Quaternion.Euler(90f, 0f, 0f);
        Vector3    startHipsPos = _body.hips.localPosition;
        while (t < 1f)
        {
            t += Time.deltaTime * 1.5f;
            float s = Mathf.SmoothStep(0,1,t);
            _body.hips.rotation      = Quaternion.Slerp(startHips, supine, s);
            _body.hips.localPosition = Vector3.Lerp(startHipsPos, new Vector3(0f,0.05f,0f), s);
            _body.leftThigh.localRotation  = Quaternion.Slerp(_body.leftThigh.localRotation,
                                                               Quaternion.Euler(0f,0f,5f), s);
            yield return null;
        }
        if (_body?.leftUpperArm  != null) _body.leftUpperArm.localEulerAngles  = new Vector3(0f,0f,-25f);
        if (_body?.rightUpperArm != null) _body.rightUpperArm.localEulerAngles = new Vector3(0f,0f, 25f);
        if (_body?.head          != null) _body.head.localEulerAngles          = Vector3.zero;
    }
}