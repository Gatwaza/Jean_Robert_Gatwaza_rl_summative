// HumanoidRescuer.cs (v3 — hands reach patient, wrist joint, IK fixes)
// ======================================================================
// Key changes:
//   - IK now targets leftWrist/rightWrist (not hand) so fingers follow
//   - Compression: both wrists driven to patient.ChestWorldPosition with
//     depth modulation — visually hands press onto the chest
//   - All IK is rotation-only (no world-position setting on hands)
//   - Wrist lock for compressions: elbows straight, wrists flat

using System.Collections;
using UnityEngine;

public class HumanoidRescuer : MonoBehaviour
{
    [Header("References")]
    public HumanoidPatient patient;

    [Header("Appearance")]
    public Color skinColor    = new Color(0.87f, 0.72f, 0.56f);
    public Color shirtColor   = new Color(0.22f, 0.42f, 0.74f);
    public Color trouserColor = new Color(0.20f, 0.20f, 0.25f);

    [Header("IK")]
    public float ikSpeed    = 9f;
    public float restDamp   = 5f;

    private HumanoidBuilder.Body _body;
    private Vector3    _leftTarget, _rightTarget;
    private bool       _ikActive    = false;
    private bool       _compressing = false;
    private float      _compPhase   = 0f;
    private Coroutine  _anim;

    void Awake()
    {
        _body = HumanoidBuilder.Build(transform, skinColor, shirtColor, trouserColor, 1f);
        AddHiVis();
        SetKneeling();
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
        if (_compressing)
            DoCompressionIK();
        else if (_ikActive)
            DriveArmsToward(_leftTarget, _rightTarget);
    }

    // ── Event handlers ────────────────────────────────────────────────────────
    void HandleState(StatePacket p)
    {
        if (_anim != null) StopCoroutine(_anim);
        switch (p.action)
        {
            case 0:  _anim = StartCoroutine(AnimAssess());        break;
            case 1:  _anim = StartCoroutine(AnimCall());          break;
            case 2:  _anim = StartCoroutine(AnimOpenAirway());    break;
            case 3:  _anim = StartCoroutine(AnimCheckBreath());   break;
            case 4:  BeginCompression(); break;
            case 5:  _anim = StartCoroutine(AnimRescueBreaths()); break;
            case 6:  _anim = StartCoroutine(AnimDefibrillate());  break;
            case 7:  _anim = StartCoroutine(AnimPulseCheck());    break;
            case 8:  _anim = StartCoroutine(AnimRecovery());      break;
            case 9:  _anim = StartCoroutine(AnimReposition());    break;
            case 10: _anim = StartCoroutine(AnimTiltHead());      break;
            default: EndCompression(); _anim = StartCoroutine(AnimIdle()); break;
        }
    }

    void HandleEpisodeEnd(StatePacket p) { EndCompression(); StartCoroutine(AnimIdle()); }
    void HandlePhaseChange(StatePacket p) { EndCompression(); SetKneeling(); }

    // ── Compression IK — drives wrists directly onto patient chest ────────────
    void BeginCompression()
    {
        _compressing = true; _ikActive = false; _compPhase = 0f;
        // Lock forearms and wrists straight for compression posture
        if (_body.leftForearm  != null) _body.leftForearm.localEulerAngles  = Vector3.zero;
        if (_body.rightForearm != null) _body.rightForearm.localEulerAngles = Vector3.zero;
        if (_body.leftWrist    != null) _body.leftWrist.localEulerAngles    = Vector3.zero;
        if (_body.rightWrist   != null) _body.rightWrist.localEulerAngles   = Vector3.zero;
        // Right hand stacks on top of left (interlaced grip)
        if (_body.rightHand    != null)
            _body.rightHand.localEulerAngles = new Vector3(-8f, 0f, 0f);
    }

    void EndCompression()
    {
        _compressing = false; _ikActive = false;
        if (_body.rightHand != null) _body.rightHand.localEulerAngles = Vector3.zero;
        if (_body.spine     != null) _body.spine.localEulerAngles = new Vector3(15f, 0f, 0f);
    }

    void DoCompressionIK()
    {
        // Pump at 100 bpm — sine wave depth
        _compPhase += Time.deltaTime * (100f / 60f) * Mathf.PI * 2f;
        float pump = Mathf.Abs(Mathf.Sin(_compPhase));   // 0=top, 1=bottom

        // Torso lean increases on downstroke
        if (_body.spine != null)
            _body.spine.localEulerAngles = new Vector3(Mathf.Lerp(18f, 40f, pump), 0f, 0f);

        // Target: patient chest + downward offset for pump depth
        Vector3 chestPos   = GetChestPos();
        Vector3 pumpOffset = -transform.up * (pump * 0.06f);
        Vector3 target     = chestPos + pumpOffset;

        // Drive both upper arms toward the target using IK
        SolveTwoBoneIK(_body.leftUpperArm,  _body.leftForearm,  _body.leftWrist,  target, left:true);
        SolveTwoBoneIK(_body.rightUpperArm, _body.rightForearm, _body.rightWrist, target, left:false);
    }

    // ── Action animations ─────────────────────────────────────────────────────
    IEnumerator AnimAssess()
    {
        _ikActive = true;
        _leftTarget  = GetPatientShoulder(true)  + Vector3.up * 0.04f;
        _rightTarget = GetPatientShoulder(false) + Vector3.up * 0.04f;
        yield return new WaitForSeconds(0.5f);
        for (int i = 0; i < 2; i++)
        {
            _leftTarget  += Vector3.up * 0.06f;
            _rightTarget += Vector3.up * 0.06f;
            yield return new WaitForSeconds(0.15f);
            _leftTarget  -= Vector3.up * 0.06f;
            _rightTarget -= Vector3.up * 0.06f;
            yield return new WaitForSeconds(0.15f);
        }
        yield return ReturnToRest();
    }

    IEnumerator AnimCall()
    {
        _ikActive = true;
        // Right hand raises to ear
        _rightTarget = transform.position + transform.up * 1.55f - transform.right * 0.18f;
        _leftTarget  = RestLeft();
        if (_body.neck != null) _body.neck.localEulerAngles = new Vector3(0f, -18f, 0f);
        yield return new WaitForSeconds(2.0f);
        if (_body.neck != null) _body.neck.localEulerAngles = Vector3.zero;
        yield return ReturnToRest();
    }

    IEnumerator AnimOpenAirway()
    {
        _ikActive = true;
        Vector3 headPos = GetPatientHead();
        _leftTarget  = headPos + transform.forward * 0.10f;
        _rightTarget = headPos - transform.forward * 0.10f;
        if (_body.spine != null) _body.spine.localEulerAngles = new Vector3(28f, 0f, 0f);
        yield return new WaitForSeconds(1.5f);
        if (_body.spine != null) _body.spine.localEulerAngles = new Vector3(15f, 0f, 0f);
        yield return ReturnToRest();
    }

    IEnumerator AnimCheckBreath()
    {
        if (_body.spine != null) _body.spine.localEulerAngles = new Vector3(42f, -8f, 0f);
        if (_body.neck  != null) _body.neck.localEulerAngles  = new Vector3(18f,  8f, 0f);
        yield return new WaitForSeconds(1.8f);
        if (_body.spine != null) _body.spine.localEulerAngles = new Vector3(15f, 0f, 0f);
        if (_body.neck  != null) _body.neck.localEulerAngles  = Vector3.zero;
    }

    IEnumerator AnimRescueBreaths()
    {
        EndCompression();
        if (_body.spine != null) _body.spine.localEulerAngles = new Vector3(38f, -4f, 0f);
        if (_body.neck  != null) _body.neck.localEulerAngles  = new Vector3(14f,  4f, 0f);
        for (int i = 0; i < 2; i++)
        {
            float t = 0f;
            while (t < 1f) {
                t += Time.deltaTime * 2.2f;
                if (_body.spine != null)
                    _body.spine.localEulerAngles = Vector3.Lerp(new Vector3(38f,-4f,0f),
                                                                 new Vector3(52f,-4f,0f), t);
                yield return null;
            }
            yield return new WaitForSeconds(0.9f);
            t = 0f;
            while (t < 1f) {
                t += Time.deltaTime * 2.2f;
                if (_body.spine != null)
                    _body.spine.localEulerAngles = Vector3.Lerp(new Vector3(52f,-4f,0f),
                                                                 new Vector3(38f,-4f,0f), t);
                yield return null;
            }
            yield return new WaitForSeconds(0.3f);
        }
        if (_body.spine != null) _body.spine.localEulerAngles = new Vector3(15f, 0f, 0f);
        if (_body.neck  != null) _body.neck.localEulerAngles  = Vector3.zero;
    }

    IEnumerator AnimDefibrillate()
    {
        EndCompression();
        Vector3 startPos = transform.position;
        Vector3 aedPos   = startPos + transform.right * (-1.1f);   // step away to AED stand
        yield return MoveRoot(startPos, aedPos, 0.9f);
        _ikActive = true;
        _rightTarget = aedPos + Vector3.up * 0.5f;
        _leftTarget  = RestLeft();
        yield return new WaitForSeconds(0.7f);
        yield return MoveRoot(aedPos, startPos, 0.9f);
        // Place pads
        Vector3 ct = GetChestPos();
        _leftTarget  = ct + transform.forward *  0.10f;
        _rightTarget = ct - transform.forward *  0.10f;
        yield return new WaitForSeconds(0.5f);
        // Stand clear
        _leftTarget  = transform.position + transform.up * 1.0f + transform.right *  0.5f;
        _rightTarget = transform.position + transform.up * 1.0f + transform.right * -0.5f;
        yield return new WaitForSeconds(0.7f);
        yield return ReturnToRest();
    }

    IEnumerator AnimPulseCheck()
    {
        _ikActive = true;
        _rightTarget = GetPatientHead() + Vector3.down * 0.04f + transform.forward * 0.06f;
        _leftTarget  = RestLeft();
        yield return new WaitForSeconds(1.5f);
        yield return ReturnToRest();
    }

    IEnumerator AnimReposition()
    {
        _ikActive = true;
        Vector3 ct = GetChestPos();
        _leftTarget  = ct + Vector3.up * 0.14f;
        _rightTarget = ct + Vector3.up * 0.14f;
        yield return new WaitForSeconds(0.25f);
        _leftTarget  = ct; _rightTarget = ct;
        yield return new WaitForSeconds(0.35f);
        yield return ReturnToRest();
    }

    IEnumerator AnimTiltHead()
    {
        _ikActive = true;
        Vector3 headPos = GetPatientHead();
        _leftTarget  = headPos + transform.forward *  0.09f;
        _rightTarget = headPos - transform.forward *  0.09f;
        if (_body.spine != null) _body.spine.localEulerAngles = new Vector3(30f, 0f, 0f);
        yield return new WaitForSeconds(1.2f);
        if (_body.spine != null) _body.spine.localEulerAngles = new Vector3(15f, 0f, 0f);
        yield return ReturnToRest();
    }

    IEnumerator AnimRecovery()
    {
        _ikActive = true;
        _leftTarget  = GetPatientShoulder(false);
        _rightTarget = patient != null
            ? patient.transform.position + new Vector3(0f, 0.55f, -0.15f)
            : GetChestPos() + Vector3.back * 0.15f;
        if (_body.spine != null) _body.spine.localEulerAngles = new Vector3(26f, 10f, 0f);
        yield return new WaitForSeconds(2.0f);
        if (_body.spine != null) _body.spine.localEulerAngles = new Vector3(15f, 0f, 0f);
        yield return ReturnToRest();
    }

    IEnumerator AnimIdle()
    {
        yield return ReturnToRest();
        float t = 0f;
        while (true)
        {
            t += Time.deltaTime;
            if (_body.spine != null)
                _body.spine.localEulerAngles = new Vector3(15f + Mathf.Sin(t*0.7f)*1.2f, 0f, 0f);
            yield return null;
        }
    }

    // ── IK ────────────────────────────────────────────────────────────────────
    void DriveArmsToward(Vector3 leftWorld, Vector3 rightWorld)
    {
        SolveTwoBoneIK(_body.leftUpperArm,  _body.leftForearm,  _body.leftWrist,  leftWorld,  left:true);
        SolveTwoBoneIK(_body.rightUpperArm, _body.rightForearm, _body.rightWrist, rightWorld, left:false);
    }

    void SolveTwoBoneIK(Transform upper, Transform mid, Transform end,
                        Vector3 targetWorld, bool left)
    {
        if (upper == null || mid == null || end == null) return;

        float upperLen = Vector3.Distance(upper.position, mid.position);
        float midLen   = Vector3.Distance(mid.position,   end.position);
        float total    = upperLen + midLen;

        Vector3 toTarget = targetWorld - upper.position;
        float   dist     = Mathf.Clamp(toTarget.magnitude, 0.01f, total * 0.999f);

        // Upper arm aim
        if (toTarget.sqrMagnitude > 0.0001f)
        {
            Vector3 hint  = left ? -Vector3.right + Vector3.down * 0.8f
                                  :  Vector3.right + Vector3.down * 0.8f;
            Vector3 fwd   = toTarget.normalized;
            Vector3 up    = Vector3.Cross(fwd, hint).normalized;
            if (up.sqrMagnitude < 0.001f) up = Vector3.up;
            Quaternion aim = Quaternion.LookRotation(fwd, up);
            upper.rotation = Quaternion.Slerp(upper.rotation, aim, Time.deltaTime * ikSpeed);
        }

        // Forearm bend (law of cosines)
        float cosAngle = (upperLen*upperLen + dist*dist - midLen*midLen)
                       / (2f*upperLen*dist + 0.0001f);
        float angle    = Mathf.Acos(Mathf.Clamp(cosAngle, -1f, 1f)) * Mathf.Rad2Deg;
        Vector3 axis   = left ? Vector3.forward : Vector3.back;
        mid.localRotation = Quaternion.Slerp(mid.localRotation,
                                              Quaternion.AngleAxis(angle, axis),
                                              Time.deltaTime * ikSpeed);
    }

    IEnumerator ReturnToRest()
    {
        _ikActive = false;
        float t = 0f;
        Quaternion luS = _body.leftUpperArm.localRotation,  ruS = _body.rightUpperArm.localRotation;
        Quaternion lfS = _body.leftForearm.localRotation,   rfS = _body.rightForearm.localRotation;
        Quaternion lwS = _body.leftWrist.localRotation,     rwS = _body.rightWrist.localRotation;

        Quaternion luR = Quaternion.Euler(15f,0f,-20f), ruR = Quaternion.Euler(15f,0f, 20f);
        Quaternion lfR = Quaternion.Euler(20f,0f,0f),   rfR = Quaternion.Euler(20f,0f, 0f);
        Quaternion lwR = Quaternion.identity,            rwR = Quaternion.identity;

        while (t < 1f)
        {
            t += Time.deltaTime * restDamp;
            float s = Mathf.SmoothStep(0,1,t);
            _body.leftUpperArm.localRotation  = Quaternion.Slerp(luS, luR, s);
            _body.rightUpperArm.localRotation = Quaternion.Slerp(ruS, ruR, s);
            _body.leftForearm.localRotation   = Quaternion.Slerp(lfS, lfR, s);
            _body.rightForearm.localRotation  = Quaternion.Slerp(rfS, rfR, s);
            _body.leftWrist.localRotation     = Quaternion.Slerp(lwS, lwR, s);
            _body.rightWrist.localRotation    = Quaternion.Slerp(rwS, rwR, s);
            yield return null;
        }
    }

    // ── Pose helpers ──────────────────────────────────────────────────────────
    void SetKneeling()
    {
        if (_body == null) return;
        if (_body.hips       != null) _body.hips.localPosition      = new Vector3(0f, 0.55f, 0f);
        if (_body.leftThigh  != null) _body.leftThigh.localEulerAngles  = new Vector3(-80f,0f, 8f);
        if (_body.rightThigh != null) _body.rightThigh.localEulerAngles = new Vector3(-80f,0f,-8f);
        if (_body.leftShin   != null) _body.leftShin.localEulerAngles   = new Vector3( 80f,0f, 0f);
        if (_body.rightShin  != null) _body.rightShin.localEulerAngles  = new Vector3( 80f,0f, 0f);
        if (_body.spine      != null) _body.spine.localEulerAngles      = new Vector3( 15f,0f, 0f);
    }

    IEnumerator MoveRoot(Vector3 from, Vector3 to, float dur)
    {
        float t = 0f;
        while (t < 1f)
        {
            t += Time.deltaTime / dur;
            transform.position = Vector3.Lerp(from, to, Mathf.SmoothStep(0,1,t));
            yield return null;
        }
    }

    // ── Position helpers ──────────────────────────────────────────────────────
    Vector3 GetChestPos() =>
        patient != null ? patient.ChestWorldPosition
                        : transform.position + transform.forward * 0.55f + Vector3.up * 0.05f;

    Vector3 GetPatientHead() =>
        patient != null ? patient.transform.position + new Vector3(0f, 0.22f, 0.82f)
                        : transform.position + transform.forward * 0.9f + Vector3.up * 0.2f;

    Vector3 GetPatientShoulder(bool left) =>
        patient != null ? patient.transform.position + new Vector3(left ? 0.20f : -0.20f, 0.24f, 0.30f)
                        : transform.position + transform.forward * 0.5f + Vector3.up * 0.2f;

    Vector3 RestLeft() =>
        _body.leftForearm != null
            ? _body.leftForearm.position + _body.leftForearm.forward * 0.20f
            : transform.position + Vector3.up * 0.7f;

    void AddHiVis()
    {
        var stripe = GameObject.CreatePrimitive(PrimitiveType.Cube);
        stripe.name = "HiVisStripe";
        stripe.transform.SetParent(_body.chest, false);
        stripe.transform.localScale    = new Vector3(0.30f, 0.03f, 0.18f);
        stripe.transform.localPosition = new Vector3(0f, 0.05f, 0.08f);
        var r = stripe.GetComponent<Renderer>();
        var m = new Material(Shader.Find("Standard"));
        m.color = new Color(1f, 0.85f, 0f);
        m.SetFloat("_Glossiness", 0.4f);
        r.material = m;
        var col = stripe.GetComponent<Collider>();
        if (col) Object.Destroy(col);
    }
}