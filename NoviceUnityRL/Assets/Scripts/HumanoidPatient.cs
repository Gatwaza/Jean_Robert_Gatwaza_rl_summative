// HumanoidPatient.cs
// ====================
// Full humanoid patient lying supine on the floor mat.
// Procedurally animates every CPR-relevant body response:
//   - Chest sink + recoil on compressions (spring physics)
//   - Head/neck rotation for airway opening
//   - Arm flop and repositioning
//   - Leg placement and knee bend
//   - Whole-body lateral roll to recovery position
//   - ROSC: subtle breathing rise, eyelid flutter (head bob), arm reach
//   - Intercept response: chest depresses realistically when hands make contact
//
// Attach to an empty GameObject named "Patient".
// No external assets or Animation Rigging package required.

using System.Collections;
using UnityEngine;

[RequireComponent(typeof(BoxCollider))]
public class HumanoidPatient : MonoBehaviour
{
    [Header("Appearance")]
    public Color skinColor  = new Color(0.65f, 0.50f, 0.38f);
    public Color shirtColor = new Color(0.85f, 0.85f, 0.85f);  // white shirt / hospital gown
    public Color trouserColor = new Color(0.35f, 0.35f, 0.40f);

    [Header("Compression Physics")]
    [Tooltip("How far the chest depresses in world units at full compression depth")]
    public float maxCompressionDepth = 0.07f;
    [Tooltip("Spring stiffness for chest recoil")]
    public float springStiffness = 18f;
    [Tooltip("Spring damping")]
    public float springDamping  = 5f;

    // ── Runtime state ────────────────────────────────────────────────────────
    private HumanoidBuilder.Body _body;
    private float _compressionTarget  = 0f;
    private float _compressionCurrent = 0f;
    private float _compressionVel     = 0f;
    private float _headTiltTarget     = 0f;
    private float _headTiltCurrent    = 0f;
    private bool  _inRecovery         = false;
    private bool  _rosc               = false;
    private float _breathCycle        = 0f;

    // Chest bone world position cache (used for IK target by rescuer)
    private Transform _chestTransform;
    public Vector3 ChestWorldPosition =>
        _chestTransform != null ? _chestTransform.position : transform.position + Vector3.up * 0.3f;

    // ── Unity lifecycle ───────────────────────────────────────────────────────
    void Awake()
    {
        // Patient lies supine on the floor mat.
        // Root stays upright (identity) — we lift it 14cm above ground
        // so body thickness doesn't clip the floor plane.
        transform.rotation = Quaternion.identity;
        transform.position = new Vector3(
            transform.position.x, 0.14f, transform.position.z);

        _body = HumanoidBuilder.Build(transform, skinColor, shirtColor, trouserColor, 1f);

        // Rotate hips 90° forward so the entire spine chain lies along world Z
        // (head toward +Z, feet toward -Z) — supine position
        _body.hips.localEulerAngles = new Vector3(90f, 0f, 0f);
        // Hips close to ground level
        _body.hips.localPosition = new Vector3(0, 0.05f, 0);

        // Arms rest at sides
        _body.leftUpperArm.localEulerAngles  = new Vector3(0, 0, -30f);
        _body.rightUpperArm.localEulerAngles = new Vector3(0, 0,  30f);
        _body.leftForearm.localEulerAngles   = new Vector3(0, 0, -15f);
        _body.rightForearm.localEulerAngles  = new Vector3(0, 0,  15f);

        // Legs flat
        _body.leftThigh.localEulerAngles  = new Vector3(0, 0, 5f);
        _body.rightThigh.localEulerAngles = new Vector3(0, 0,-5f);

        // Ground collider so rescuer can walk around
        var col = GetComponent<BoxCollider>();
        col.size   = new Vector3(0.5f, 0.25f, 1.8f);
        col.center = new Vector3(0, 0.12f, 0);

        _chestTransform = _body.chest;

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

    // ── Event handlers ────────────────────────────────────────────────────────
    void HandleState(StatePacket p)
    {
        switch (p.action)
        {
            case 4:  // BEGIN_CHEST_COMPRESSIONS
                float depth = p.vitals?.hand_placement ?? 0.5f;
                _compressionTarget = maxCompressionDepth * Mathf.Clamp01(0.4f + depth * 0.6f);
                break;
            case 2: case 10: // OPEN_AIRWAY / TILT_HEAD_BACK
                _headTiltTarget = 35f;
                break;
            case 5:  // RESCUE_BREATHS — trigger chest rise
                StartCoroutine(ChestRise());
                break;
            case 8:  // RECOVERY_POSITION
                if (!_inRecovery) StartCoroutine(RollToRecovery());
                break;
            default:
                _compressionTarget = 0f;
                break;
        }
        _rosc = p.rosc;
        if (_rosc && !_inRecovery) StartCoroutine(ROSCResponse());
    }

    void HandleEpisodeEnd(StatePacket p)
    {
        StopAllCoroutines();
        StartCoroutine(ResetToSupine());
    }

    void HandlePhaseChange(StatePacket p)
    {
        StopAllCoroutines();
        StartCoroutine(ResetToSupine());
    }

    // ── Per-frame animations ──────────────────────────────────────────────────

    void AnimateChestSpring()
    {
        // Spring physics: chest bone moves toward _compressionTarget
        float error = _compressionTarget - _compressionCurrent;
        _compressionVel += (error * springStiffness - _compressionVel * springDamping)
                           * Time.deltaTime;
        _compressionCurrent += _compressionVel * Time.deltaTime;
        _compressionCurrent  = Mathf.Clamp(_compressionCurrent, 0f, maxCompressionDepth);

        // Apply as local Y offset on chest (patient is rotated so Y = depth into body)
        var lp = _body.chest.localPosition;
        _body.chest.localPosition = new Vector3(lp.x, lp.y, -_compressionCurrent);

        // Shoulders follow chest slightly — realistic torso flex
        float shoulderFollow = _compressionCurrent * 0.4f;
        _body.leftShoulder.localPosition  = new Vector3(
            _body.leftShoulder.localPosition.x,
            _body.leftShoulder.localPosition.y,
            -shoulderFollow);
        _body.rightShoulder.localPosition = new Vector3(
            _body.rightShoulder.localPosition.x,
            _body.rightShoulder.localPosition.y,
            -shoulderFollow);
    }

    void AnimateHeadTilt()
    {
        _headTiltCurrent = Mathf.MoveTowards(
            _headTiltCurrent, _headTiltTarget, 60f * Time.deltaTime);
        // Neck extends (X rotation in supine = head tilts back)
        _body.neck.localEulerAngles = new Vector3(_headTiltCurrent, 0, 0);
        // Head follows at half the angle
        _body.head.localEulerAngles = new Vector3(_headTiltCurrent * 0.5f, 0, 0);
    }

    void AnimateBreathing()
    {
        if (!_rosc && _compressionTarget < 0.01f) return;

        // After ROSC: visible chest rise with breathing cycle
        _breathCycle += Time.deltaTime * (_rosc ? 0.4f : 0.2f);
        float rise = Mathf.Max(0, Mathf.Sin(_breathCycle * Mathf.PI * 2f)) * 0.015f;
        var lp = _body.chest.localPosition;
        _body.chest.localPosition = new Vector3(lp.x, lp.y + rise * 0.01f, lp.z);
    }

    // ── Coroutine animations ──────────────────────────────────────────────────

    IEnumerator ChestRise()
    {
        // Two rescue breaths — chest rises and falls twice
        for (int i = 0; i < 2; i++)
        {
            float t = 0f;
            while (t < 1f) {
                t += Time.deltaTime * 1.5f;
                float rise = Mathf.Sin(t * Mathf.PI) * 0.025f;
                var lp = _body.chest.localPosition;
                _body.chest.localPosition = new Vector3(lp.x, lp.y, lp.z - rise);
                yield return null;
            }
            yield return new WaitForSeconds(0.3f);
        }
    }

    IEnumerator RollToRecovery()
    {
        _inRecovery = true;
        float t = 0f;
        Quaternion startRot = _body.hips.rotation;
        Quaternion targetRot = startRot * Quaternion.Euler(0, 0, 75f);

        // Also bend top knee (left leg) for stable recovery position
        Quaternion kneeBend = Quaternion.Euler(0, 0, -60f);
        Quaternion kneeStart = _body.leftThigh.localRotation;

        while (t < 1f)
        {
            t += Time.deltaTime * 0.6f;
            float smooth = Mathf.SmoothStep(0, 1, t);
            _body.hips.rotation = Quaternion.Slerp(startRot, targetRot, smooth);
            _body.leftThigh.localRotation =
                Quaternion.Slerp(kneeStart, kneeBend, smooth);
            yield return null;
        }
    }

    IEnumerator ROSCResponse()
    {
        yield return new WaitForSeconds(0.5f);

        // Subtle head turn — patient becoming aware
        float t = 0f;
        while (t < 1f)
        {
            t += Time.deltaTime * 0.8f;
            float nod = Mathf.Sin(t * Mathf.PI * 2f) * 8f;
            _body.head.localEulerAngles = new Vector3(
                _body.head.localEulerAngles.x, nod, 0);
            yield return null;
        }

        // Arm slight raise — reaching for consciousness
        t = 0f;
        Quaternion armStart  = _body.rightUpperArm.localRotation;
        Quaternion armTarget = armStart * Quaternion.Euler(-30f, 0, 0);
        while (t < 1f)
        {
            t += Time.deltaTime * 1.2f;
            _body.rightUpperArm.localRotation =
                Quaternion.Slerp(armStart, armTarget, Mathf.SmoothStep(0, 1, t));
            yield return null;
        }
        // Return arm
        t = 0f;
        while (t < 1f)
        {
            t += Time.deltaTime * 0.6f;
            _body.rightUpperArm.localRotation =
                Quaternion.Slerp(armTarget, armStart, Mathf.SmoothStep(0, 1, t));
            yield return null;
        }

        // Breathing begins: handled in AnimateBreathing() via _rosc flag
    }

    IEnumerator ResetToSupine()
    {
        _inRecovery = false;
        _rosc = false;
        _compressionTarget  = 0f;
        _compressionCurrent = 0f;
        _compressionVel     = 0f;
        _headTiltTarget     = 0f;
        _breathCycle        = 0f;

        float t = 0f;
        Quaternion startHips = _body.hips.rotation;
        Quaternion baseHips  = Quaternion.Euler(90f, 0f, 0f);
        while (t < 1f)
        {
            t += Time.deltaTime * 1.5f;
            _body.hips.rotation = Quaternion.Slerp(startHips, baseHips,
                                                    Mathf.SmoothStep(0, 1, t));
            _body.hips.localPosition = new Vector3(0, 0.05f, 0);
            _body.leftThigh.localRotation = Quaternion.Slerp(
                _body.leftThigh.localRotation, Quaternion.Euler(0,0,5f),
                Mathf.SmoothStep(0, 1, t));
            yield return null;
        }

        // Reset chest local position
        _body.chest.localPosition = new Vector3(0, 0.20f, 0);
    }
}
