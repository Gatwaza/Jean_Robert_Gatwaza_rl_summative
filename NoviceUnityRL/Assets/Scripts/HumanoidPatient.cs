// HumanoidPatient.cs (v5 — spine-only ROSC sit-up, legs fixed on mat, auto lie-back per episode)
//
// Root cause of v4 bugs:
//   1. SitUpROSC moved _body.hips.localPosition by +0.43 units AND rotated hips 68°
//      simultaneously — this pushed the entire body hierarchy through the floor and
//      caused child transforms to fly apart (the "head off" visual).
//   2. Rotating hips from Euler(90,0,0) toward Euler(22,0,0) rotated the legs
//      (children of hips) downward into the floor.
//   3. No automatic lie-back: the patient stayed seated until the next episode reset.
//
// v5 fixes:
//   • SitUpROSC rotates SPINE only — hips remain at Euler(90,0,0), legs stay on mat.
//     In the supine rig, spine local-X == world X; rotating spine -X tilts the
//     upper body upward (toward seated) while hips/legs never move.
//   • After the sit-up, patient automatically lies back down smoothly.
//   • HandleEpisodeEnd/HandlePhaseChange stop any in-flight SitUpROSC coroutine
//     before calling ResetPose, preventing coroutine conflicts.
//   • ResetPose now resets spine and chest local rotations as well.
//   • Hips base localPosition standardised to (0, 0.08, 0) everywhere.

using System.Collections;
using UnityEngine;

[RequireComponent(typeof(BoxCollider))]
public class HumanoidPatient : MonoBehaviour
{
    [Header("Appearance")]
    public Color skinColor    = new Color(0.65f, 0.50f, 0.38f);
    public Color shirtColor   = new Color(0.88f, 0.88f, 0.88f);
    public Color trouserColor = new Color(0.32f, 0.32f, 0.38f);

    [Header("Compression Spring")]
    public float maxDepth = 0.055f;
    public float stiffness = 22f;
    public float damping   = 7f;

    // ── Private state ─────────────────────────────────────────────────────────
    private HumanoidBuilder.Body _body;

    private float _compTarget  = 0f, _compCurrent = 0f, _compVel = 0f;
    private float _headTiltTgt = 0f, _headTiltCur = 0f;
    private bool  _inRecovery  = false;
    private bool  _rosc        = false;
    private bool  _sittingUp   = false;
    private float _breathCycle = 0f;

    // Reference kept so we can stop it cleanly on episode end
    private Coroutine _sitUpCoroutine;

    // ── Supine chest position (used by HumanoidRescuer for IK) ───────────────
    public Vector3 ChestWorldPosition =>
        transform.position + new Vector3(0f, 0.22f, 0.10f);

    // ── Base hips local position in supine pose ───────────────────────────────
    private static readonly Vector3 HipsBaseLocalPos = new Vector3(0f, 0.08f, 0f);

    // ── Awake ─────────────────────────────────────────────────────────────────
    void Awake()
    {
        transform.rotation = Quaternion.identity;
        transform.position = new Vector3(transform.position.x, 0.19f, transform.position.z);

        _body = HumanoidBuilder.Build(transform, skinColor, shirtColor, trouserColor, 1f);

        // Supine: rotate hips 90° on X so the body lies flat along world Z
        _body.hips.localEulerAngles = new Vector3(90f, 0f, 0f);
        _body.hips.localPosition    = HipsBaseLocalPos;

        // Arms rest at sides
        _body.leftUpperArm.localEulerAngles  = new Vector3(0f, 0f, -20f);
        _body.rightUpperArm.localEulerAngles = new Vector3(0f, 0f,  20f);
        _body.leftForearm.localEulerAngles   = new Vector3(0f, 0f, -8f);
        _body.rightForearm.localEulerAngles  = new Vector3(0f, 0f,  8f);

        // Legs parallel, slight outward splay — no crossing
        _body.leftThigh.localEulerAngles  = new Vector3(0f, 0f,  8f);
        _body.rightThigh.localEulerAngles = new Vector3(0f, 0f, -8f);
        _body.leftShin.localEulerAngles   = Vector3.zero;
        _body.rightShin.localEulerAngles  = Vector3.zero;

        var col    = GetComponent<BoxCollider>();
        col.size   = new Vector3(0.5f, 0.25f, 1.8f);
        col.center = new Vector3(0f, 0.12f, 0f);

        BridgeEvents.OnStateUpdate += HandleState;
        BridgeEvents.OnEpisodeEnd  += HandleEpisodeEnd;
        BridgeEvents.OnPhaseChange += HandlePhaseChange;
    }

    void OnDestroy()
    {
        BridgeEvents.OnStateUpdate -= HandleState;
        BridgeEvents.OnEpisodeEnd  -= HandleEpisodeEnd;
        BridgeEvents.OnPhaseChange -= HandlePhaseChange;
    }

    // ── Update ────────────────────────────────────────────────────────────────
    void Update()
    {
        AnimateChestSpring();
        AnimateHeadTilt();
        AnimateBreathing();
    }

    // ── Bridge event handlers ─────────────────────────────────────────────────
    void HandleState(StatePacket p)
    {
        switch (p.action)
        {
            case 4: // BEGIN_CHEST_COMPRESSIONS
                float depth = Mathf.Clamp01(0.4f + (p.vitals?.hand_placement ?? 0.5f) * 0.6f);
                _compTarget = maxDepth * depth;
                break;
            case 2:  // OPEN_AIRWAY
            case 10: // TILT_HEAD_BACK
                _headTiltTgt = 30f;
                break;
            case 5: // DELIVER_RESCUE_BREATHS
                StartCoroutine(ChestRiseTwice());
                break;
            case 8: // RECOVERY_POSITION
                if (!_inRecovery) StartCoroutine(GentleRecoveryTilt());
                break;
            default:
                _compTarget = 0f;
                break;
        }

        if (p.rosc && !_sittingUp)
        {
            _rosc = true;
            _sitUpCoroutine = StartCoroutine(SitUpROSC());
        }
    }

    void HandleEpisodeEnd(StatePacket p)
    {
        // Cancel any in-flight ROSC animation so it doesn't fight ResetPose
        StopSitUp();
        StartCoroutine(ResetPose());
    }

    void HandlePhaseChange(StatePacket p)
    {
        StopSitUp();
        StartCoroutine(ResetPose());
    }

    void StopSitUp()
    {
        if (_sitUpCoroutine != null)
        {
            StopCoroutine(_sitUpCoroutine);
            _sitUpCoroutine = null;
        }
        _sittingUp = false;
        _rosc      = false;
    }

    // ── Per-frame animations ──────────────────────────────────────────────────
    void AnimateChestSpring()
    {
        float err = _compTarget - _compCurrent;
        _compVel     += (err * stiffness - _compVel * damping) * Time.deltaTime;
        _compCurrent  = Mathf.Clamp(_compCurrent + _compVel * Time.deltaTime, 0f, maxDepth);

        if (_body?.chest != null)
        {
            var lp = _body.chest.localPosition;
            _body.chest.localPosition = new Vector3(lp.x, lp.y, -_compCurrent);
        }
        float sf = _compCurrent * 0.3f;
        if (_body?.leftShoulder  != null) { var lp = _body.leftShoulder.localPosition;
            _body.leftShoulder.localPosition  = new Vector3(lp.x, lp.y, -sf); }
        if (_body?.rightShoulder != null) { var lp = _body.rightShoulder.localPosition;
            _body.rightShoulder.localPosition = new Vector3(lp.x, lp.y, -sf); }
    }

    void AnimateHeadTilt()
    {
        // Only drive head tilt when not in ROSC sit-up animation
        if (_sittingUp) return;
        _headTiltCur = Mathf.MoveTowards(_headTiltCur, _headTiltTgt, 50f * Time.deltaTime);
        if (_body?.neck != null) _body.neck.localEulerAngles = new Vector3(_headTiltCur, 0f, 0f);
        if (_body?.head != null) _body.head.localEulerAngles = new Vector3(_headTiltCur * 0.4f, 0f, 0f);
    }

    void AnimateBreathing()
    {
        if (!_rosc) return;
        _breathCycle += Time.deltaTime * 0.3f;
        float rise = Mathf.Max(0f, Mathf.Sin(_breathCycle * Mathf.PI * 2f)) * 0.010f;
        if (_body?.chest != null)
        {
            var lp = _body.chest.localPosition;
            _body.chest.localPosition = new Vector3(lp.x, lp.y + rise, lp.z);
        }
    }

    // ── Recovery position (action 8) — gentle lateral tilt ───────────────────
    IEnumerator GentleRecoveryTilt()
    {
        _inRecovery = true;
        Quaternion hipsStart  = _body.hips.localRotation;
        Quaternion hipsTilted = Quaternion.Euler(90f, 0f, 35f);
        Quaternion kneeStart  = _body.leftShin.localRotation;
        Quaternion kneeBend   = Quaternion.Euler(0f, 0f, -22f);
        float t = 0f;
        while (t < 1f)
        {
            t += Time.deltaTime * 0.7f;
            float s = Mathf.SmoothStep(0f, 1f, t);
            _body.hips.localRotation     = Quaternion.Slerp(hipsStart, hipsTilted, s);
            _body.leftShin.localRotation = Quaternion.Slerp(kneeStart, kneeBend,   s);
            yield return null;
        }
    }

    // ── ROSC sit-up (v5) ──────────────────────────────────────────────────────
    //
    // KEY INSIGHT: Only the SPINE is rotated — hips stay at Euler(90,0,0).
    //
    //   In the supine rig hips.localEulerAngles = (90,0,0), which means:
    //     spine local-X  ==  world X  (no change)
    //     spine local-Y  ==  world +Z (head direction along floor)
    //     spine local-Z  ==  world -Y (downward)
    //
    //   Rotating spine around its local X by -65° tilts the local Y axis (world Z)
    //   up toward world +Y, raising the torso to roughly a seated angle — while the
    //   hips and all leg bones remain completely untouched on the mat.
    //
    //   After a short hold the coroutine automatically lays the patient back down,
    //   returning to the supine pose ready for the next episode.
    //
    IEnumerator SitUpROSC()
    {
        _sittingUp = true;

        // Freeze head-tilt drive so AnimateHeadTilt doesn't fight the nod animation
        _headTiltTgt = 0f;
        _headTiltCur = 0f;

        yield return new WaitForSeconds(0.5f);

        // ── Capture resting spine / chest rotations ───────────────────────────
        Quaternion spineRest  = _body.spine.localRotation;   // Quaternion.identity
        Quaternion chestRest  = _body.chest.localRotation;   // Quaternion.identity

        // ── Target: spine tilted -65° on local X ≈ seated upright ────────────
        // Chest adds a small extra lean for a natural "propped up" look
        Quaternion spineSitUp = Quaternion.Euler(-65f, 0f, 0f);
        Quaternion chestSitUp = Quaternion.Euler(-12f, 0f, 0f);

        // ── Phase 1: Rise (~3 s) ──────────────────────────────────────────────
        float t = 0f;
        while (t < 1f)
        {
            t += Time.deltaTime * 0.34f;
            float s = Mathf.SmoothStep(0f, 1f, t);
            if (_body?.spine != null) _body.spine.localRotation = Quaternion.Slerp(spineRest, spineSitUp, s);
            if (_body?.chest != null) _body.chest.localRotation = Quaternion.Slerp(chestRest, chestSitUp, s);
            yield return null;
        }

        // ── Phase 2: Head bobs — regaining consciousness (~1.3 s) ────────────
        t = 0f;
        while (t < 1f)
        {
            t += Time.deltaTime * 0.75f;
            float nod = Mathf.Sin(t * Mathf.PI * 2.5f) * 7f;
            if (_body?.head != null)
                _body.head.localEulerAngles = new Vector3(nod, nod * 0.12f, 0f);
            yield return null;
        }
        // Settle head
        if (_body?.head != null) _body.head.localEulerAngles = Vector3.zero;
        if (_body?.neck != null) _body.neck.localEulerAngles = Vector3.zero;

        // Arms lift slightly — natural recovery posture
        if (_body?.leftUpperArm  != null) _body.leftUpperArm.localEulerAngles  = new Vector3(-12f, 0f, -10f);
        if (_body?.rightUpperArm != null) _body.rightUpperArm.localEulerAngles = new Vector3(-12f, 0f,  10f);

        // ── Phase 3: Hold seated (1.5 s) ─────────────────────────────────────
        yield return new WaitForSeconds(1.5f);

        // ── Phase 4: Lie back down (~2.4 s) ──────────────────────────────────
        Quaternion spineCurrent = _body.spine != null ? _body.spine.localRotation : spineSitUp;
        Quaternion chestCurrent = _body.chest != null ? _body.chest.localRotation : chestSitUp;

        t = 0f;
        while (t < 1f)
        {
            t += Time.deltaTime * 0.42f;
            float s = Mathf.SmoothStep(0f, 1f, t);

            if (_body?.spine != null) _body.spine.localRotation = Quaternion.Slerp(spineCurrent, spineRest, s);
            if (_body?.chest != null) _body.chest.localRotation = Quaternion.Slerp(chestCurrent, chestRest, s);

            // Arms return to resting position
            if (_body?.leftUpperArm  != null)
                _body.leftUpperArm.localEulerAngles  = Vector3.Lerp(new Vector3(-12f, 0f, -10f),
                                                                      new Vector3(  0f, 0f, -20f), s);
            if (_body?.rightUpperArm != null)
                _body.rightUpperArm.localEulerAngles = Vector3.Lerp(new Vector3(-12f, 0f,  10f),
                                                                      new Vector3(  0f, 0f,  20f), s);
            yield return null;
        }

        // ── Fully supine again — clean up ─────────────────────────────────────
        if (_body?.head != null) _body.head.localEulerAngles = Vector3.zero;
        if (_body?.neck != null) _body.neck.localEulerAngles = Vector3.zero;

        _rosc      = false;   // stops AnimateBreathing
        _sittingUp = false;
        _sitUpCoroutine = null;
        _breathCycle    = 0f;
    }

    // ── Rescue breaths — chest rises twice ───────────────────────────────────
    IEnumerator ChestRiseTwice()
    {
        for (int i = 0; i < 2; i++)
        {
            float t = 0f;
            while (t < 1f)
            {
                t += Time.deltaTime * 1.8f;
                float rise = Mathf.Sin(t * Mathf.PI) * 0.018f;
                if (_body?.chest != null)
                {
                    var lp = _body.chest.localPosition;
                    _body.chest.localPosition = new Vector3(lp.x, lp.y, lp.z - rise);
                }
                yield return null;
            }
            yield return new WaitForSeconds(0.2f);
        }
    }

    // ── Full pose reset (called on episode end and phase change) ──────────────
    IEnumerator ResetPose()
    {
        // Clear all state flags
        _sittingUp   = false;
        _rosc        = false;
        _inRecovery  = false;
        _compTarget  = 0f;
        _compCurrent = 0f;
        _compVel     = 0f;
        _headTiltTgt = 0f;
        _headTiltCur = 0f;
        _breathCycle = 0f;

        // Snapshot current poses for smooth interpolation back to supine
        Quaternion hipsStart  = _body.hips  != null ? _body.hips.localRotation  : Quaternion.identity;
        Vector3    hipsPosNow = _body.hips  != null ? _body.hips.localPosition  : HipsBaseLocalPos;
        Quaternion spineNow   = _body.spine != null ? _body.spine.localRotation : Quaternion.identity;
        Quaternion chestNow   = _body.chest != null ? _body.chest.localRotation : Quaternion.identity;
        Quaternion neckNow    = _body.neck  != null ? _body.neck.localRotation  : Quaternion.identity;
        Quaternion headNow    = _body.head  != null ? _body.head.localRotation  : Quaternion.identity;

        Quaternion hipsSupine = Quaternion.Euler(90f, 0f, 0f);

        float t = 0f;
        while (t < 1f)
        {
            t += Time.deltaTime * 1.4f;
            float s = Mathf.SmoothStep(0f, 1f, t);

            if (_body?.hips  != null) _body.hips.localRotation  = Quaternion.Slerp(hipsStart, hipsSupine,          s);
            if (_body?.hips  != null) _body.hips.localPosition  = Vector3.Lerp(hipsPosNow, HipsBaseLocalPos,         s);
            if (_body?.spine != null) _body.spine.localRotation = Quaternion.Slerp(spineNow, Quaternion.identity,   s);
            if (_body?.chest != null) _body.chest.localRotation = Quaternion.Slerp(chestNow, Quaternion.identity,   s);
            if (_body?.neck  != null) _body.neck.localRotation  = Quaternion.Slerp(neckNow,  Quaternion.identity,   s);
            if (_body?.head  != null) _body.head.localRotation  = Quaternion.Slerp(headNow,  Quaternion.identity,   s);
            yield return null;
        }

        // Hard-set all limb poses to supine defaults
        if (_body?.leftUpperArm  != null) _body.leftUpperArm.localEulerAngles  = new Vector3(0f, 0f, -20f);
        if (_body?.rightUpperArm != null) _body.rightUpperArm.localEulerAngles = new Vector3(0f, 0f,  20f);
        if (_body?.leftForearm   != null) _body.leftForearm.localEulerAngles   = new Vector3(0f, 0f,  -8f);
        if (_body?.rightForearm  != null) _body.rightForearm.localEulerAngles  = new Vector3(0f, 0f,   8f);
        if (_body?.head          != null) _body.head.localEulerAngles          = Vector3.zero;
        if (_body?.neck          != null) _body.neck.localEulerAngles          = Vector3.zero;
        if (_body?.spine         != null) _body.spine.localRotation            = Quaternion.identity;
        if (_body?.chest         != null) _body.chest.localRotation            = Quaternion.identity;
        if (_body?.leftShin      != null) _body.leftShin.localRotation         = Quaternion.identity;
        if (_body?.rightShin     != null) _body.rightShin.localRotation        = Quaternion.identity;
        if (_body?.leftThigh     != null) _body.leftThigh.localEulerAngles     = new Vector3(0f, 0f,  8f);
        if (_body?.rightThigh    != null) _body.rightThigh.localEulerAngles    = new Vector3(0f, 0f, -8f);
        if (_body?.leftFoot      != null) _body.leftFoot.localRotation         = Quaternion.identity;
        if (_body?.rightFoot     != null) _body.rightFoot.localRotation        = Quaternion.identity;
    }
}