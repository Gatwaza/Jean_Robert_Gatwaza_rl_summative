// HumanoidPatient.cs (v4 — legs flat/parallel, no crossing, gentle recovery tilt, sit-up on ROSC)
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
    public float maxDepth     = 0.055f;
    public float stiffness    = 22f;
    public float damping      = 7f;

    private HumanoidBuilder.Body _body;
    private float _compTarget  = 0f, _compCurrent = 0f, _compVel = 0f;
    private float _headTiltTgt = 0f, _headTiltCur = 0f;
    private bool  _inRecovery  = false;
    private bool  _rosc        = false;
    private bool  _sittingUp   = false;
    private float _breathCycle = 0f;

    // Supine patient: the hips are rotated 90° on X, so _body.chest.position
    // points *above* the patient rather than at sternum height above the floor.
    // We compute the IK target directly from the root transform instead.
    public Vector3 ChestWorldPosition =>
        transform.position + new Vector3(0f, 0.22f, 0.10f);

    void Awake()
    {
        transform.rotation = Quaternion.identity;
        transform.position = new Vector3(transform.position.x, 0.19f, transform.position.z);

        _body = HumanoidBuilder.Build(transform, skinColor, shirtColor, trouserColor, 1f);

        // Supine: hips rotated so body lies along Z axis
        _body.hips.localEulerAngles = new Vector3(90f, 0f, 0f);
        _body.hips.localPosition = new Vector3(0f, 0.08f, 0f);

        // Arms rest at sides, flat
        _body.leftUpperArm.localEulerAngles  = new Vector3(0f, 0f, -20f);
        _body.rightUpperArm.localEulerAngles = new Vector3(0f, 0f,  20f);
        _body.leftForearm.localEulerAngles   = new Vector3(0f, 0f, -8f);
        _body.rightForearm.localEulerAngles  = new Vector3(0f, 0f,  8f);

        // FIX: legs parallel, slight outward splay — NO crossing
        // In supine with hips rotated 90° on X, leg angle Z controls left/right spread
        _body.leftThigh.localEulerAngles  = new Vector3(0f, 0f,  8f);
        _body.rightThigh.localEulerAngles = new Vector3(0f, 0f, -8f);
        _body.leftShin.localEulerAngles   = Vector3.zero;
        _body.rightShin.localEulerAngles  = Vector3.zero;

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

    void HandleState(StatePacket p)
    {
        switch (p.action)
        {
            case 4:
                float depth = Mathf.Clamp01(0.4f + (p.vitals?.hand_placement ?? 0.5f) * 0.6f);
                _compTarget = maxDepth * depth;
                break;
            case 2: case 10:
                _headTiltTgt = 30f;
                break;
            case 5:
                StartCoroutine(ChestRiseTwice());
                break;
            case 8:
                // FIX: gentle upper-body tilt only — no full body rotation
                if (!_inRecovery) StartCoroutine(GentleRecoveryTilt());
                break;
            default:
                _compTarget = 0f;
                break;
        }

        if (p.rosc && !_sittingUp)
        {
            _rosc = true;
            StartCoroutine(SitUpROSC());
        }
    }

    void HandleEpisodeEnd(StatePacket p) => StartCoroutine(ResetPose());
    void HandlePhaseChange(StatePacket p) => StartCoroutine(ResetPose());

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
        _headTiltCur = Mathf.MoveTowards(_headTiltCur, _headTiltTgt, 50f * Time.deltaTime);
        if (_body?.neck != null) _body.neck.localEulerAngles = new Vector3(_headTiltCur, 0, 0);
        if (_body?.head != null) _body.head.localEulerAngles = new Vector3(_headTiltCur * 0.4f, 0, 0);
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

    // FIX: Roll hips ~35° on Z to tilt the supine patient onto their side.
    // Rotating the spine on Z with hips at 90° X just twisted the torso
    // in the wrong axis — rolling from the root (hips) is correct.
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
            float s = Mathf.SmoothStep(0, 1, t);
            _body.hips.localRotation     = Quaternion.Slerp(hipsStart, hipsTilted, s);
            _body.leftShin.localRotation = Quaternion.Slerp(kneeStart, kneeBend,   s);
            yield return null;
        }
    }

    IEnumerator SitUpROSC()
    {
        _sittingUp = true;
        yield return new WaitForSeconds(0.6f);

        Quaternion hipsStart  = _body.hips.localRotation;
        // 22° = roughly seated-upright when built from supine 90° start
        Quaternion hipsSeated = Quaternion.Euler(22f, 0f, 0f);
        Vector3 hipsPosStart  = _body.hips.localPosition;

        // The feet need to stay on the floor (y≈0) as the body rises.
        // We interpolate hips upward AND compensate foot Z so toes don't clip.
        // In the supine rig the feet are at the -Z end of the body.
        // As the torso rises, the feet slide slightly forward (toward +Z) to
        // maintain ground contact — same as a person pushing their heels in.
        Quaternion footStart   = _body.leftFoot.localRotation;   // right mirrors this
        Quaternion footGrounded = Quaternion.Euler(-60f, 0f, 0f); // toes push into floor

        float t = 0f;
        while (t < 1f)
        {
            t += Time.deltaTime * 0.32f;
            float s = Mathf.SmoothStep(0, 1, t);

            _body.hips.localRotation = Quaternion.Slerp(hipsStart, hipsSeated, s);
            // Rise from 0.05 to 0.48 — stays well above floor surface
            _body.hips.localPosition = Vector3.Lerp(hipsPosStart, new Vector3(0f, 0.48f, 0f), s);

            // Bend knees as body rises — keeps feet anchored visually
            _body.leftThigh.localEulerAngles  = new Vector3(0f, 0f,  8f + s*30f);
            _body.rightThigh.localEulerAngles = new Vector3(0f, 0f, -8f - s*30f);
            _body.leftShin.localEulerAngles   = new Vector3(0f, 0f, -s*40f);
            _body.rightShin.localEulerAngles  = new Vector3(0f, 0f,  s*40f);

            // Feet stay flat — rotate toward floor-facing as knees bend
            _body.leftFoot.localRotation  = Quaternion.Slerp(footStart, footGrounded, s);
            _body.rightFoot.localRotation = Quaternion.Slerp(footStart, footGrounded, s);

            // World height guard: clamp entire root above y=0
            Vector3 rp = transform.position;
            if (rp.y < 0.10f) transform.position = new Vector3(rp.x, 0.10f, rp.z);

            yield return null;
        }

        // Head bobs — regaining consciousness
        t = 0f;
        while (t < 1f)
        {
            t += Time.deltaTime * 0.9f;
            float nod = Mathf.Sin(t * Mathf.PI * 1.5f) * 10f;
            if (_body?.head != null) _body.head.localEulerAngles = new Vector3(nod, nod*0.2f, 0);
            yield return null;
        }

        // Arms reach slightly forward — natural recovery posture
        if (_body?.leftUpperArm  != null) _body.leftUpperArm.localEulerAngles  = new Vector3(-15f,0f,-10f);
        if (_body?.rightUpperArm != null) _body.rightUpperArm.localEulerAngles = new Vector3(-15f,0f, 10f);
    }

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

    IEnumerator ResetPose()
    {
        _sittingUp = false; _rosc = false; _inRecovery = false;
        _compTarget = 0f; _compCurrent = 0f; _compVel = 0f;
        _headTiltTgt = 0f; _breathCycle = 0f;

        float t = 0f;
        Quaternion hipsStart  = _body.hips.localRotation;
        Quaternion hipsSupine = Quaternion.Euler(90f, 0f, 0f);
        Vector3 hipsPosStart  = _body.hips.localPosition;
        while (t < 1f)
        {
            t += Time.deltaTime * 1.4f;
            float s = Mathf.SmoothStep(0,1,t);
            _body.hips.localRotation = Quaternion.Slerp(hipsStart, hipsSupine, s);
            _body.hips.localPosition = Vector3.Lerp(hipsPosStart, new Vector3(0f,0.05f,0f), s);
            yield return null;
        }
        // Restore all poses
        if (_body?.leftUpperArm  != null) _body.leftUpperArm.localEulerAngles  = new Vector3(0f,0f,-20f);
        if (_body?.rightUpperArm != null) _body.rightUpperArm.localEulerAngles = new Vector3(0f,0f, 20f);
        if (_body?.head          != null) _body.head.localEulerAngles          = Vector3.zero;
        if (_body?.spine         != null) _body.spine.localRotation            = Quaternion.identity;
        if (_body?.leftShin      != null) _body.leftShin.localRotation         = Quaternion.identity;
        // Reset legs to no-crossing
        if (_body?.leftThigh     != null) _body.leftThigh.localEulerAngles     = new Vector3(0f,0f, 8f);
        if (_body?.rightThigh    != null) _body.rightThigh.localEulerAngles    = new Vector3(0f,0f,-8f);
    }
}