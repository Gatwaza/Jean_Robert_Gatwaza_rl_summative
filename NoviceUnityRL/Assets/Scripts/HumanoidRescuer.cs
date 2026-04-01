// HumanoidRescuer.cs  (v5 — proper kneeling, BOTH arms IK, smooth elbow, chest contact)
// Both arms always move together. Elbow uses SmoothDamp to remove snap.
// Compression: wrists reach patient.ChestWorldPosition with pump depth offset.
// Pulse check: right hand reaches neck of patient.
// Kneeling: thighs vertical, shins horizontal, feet dorsiflexed — NOT squatting.

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

    [Header("IK Settings")]
    [Tooltip("How fast arms track IK targets (degrees/s equivalent)")]
    public float ikSpeed    = 12f;
    [Tooltip("Seconds to return to rest pose")]
    public float restTime   = 0.25f;

    // ── Internal state ────────────────────────────────────────────────────────
    private HumanoidBuilder.Body _body;

    // IK targets — both arms always get a target
    private Vector3 _leftTgt, _rightTgt;
    private bool    _ikActive    = false;
    private bool    _compressing = false;
    private float   _compPhase   = 0f;

    // Per-joint velocity for SmoothDamp (elbow smoothing)
    private Quaternion _luVel = Quaternion.identity, _ruVel = Quaternion.identity;
    private Quaternion _lfVel = Quaternion.identity, _rfVel = Quaternion.identity;
    private Quaternion _lwVel = Quaternion.identity, _rwVel = Quaternion.identity;

    private Coroutine _anim;

    // ── Cache rest rotations per joint ────────────────────────────────────────
    private Quaternion _luRest, _ruRest, _lfRest, _rfRest, _lwRest, _rwRest;

    void Awake()
    {
        _body = HumanoidBuilder.Build(transform, skinColor, shirtColor, trouserColor, 1f);
        AddHiVis();
        SetKneeling();
        CacheRestPose();

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

    void CacheRestPose()
    {
        // Natural kneeling rest: arms hang at sides, slightly forward
        _luRest = Quaternion.Euler(12f,  0f, -22f);
        _ruRest = Quaternion.Euler(12f,  0f,  22f);
        _lfRest = Quaternion.Euler(18f,  0f,   0f);
        _rfRest = Quaternion.Euler(18f,  0f,   0f);
        _lwRest = Quaternion.identity;
        _rwRest = Quaternion.identity;
    }

    void Update()
    {
        if (_compressing)
            DoCompressionIK();
        else if (_ikActive)
            SolveArms(_leftTgt, _rightTgt);
        else
            DriftToRest();
    }

    // ── Event handler ─────────────────────────────────────────────────────────
    void HandleState(StatePacket p)
    {
        if (_anim != null) StopCoroutine(_anim);
        EndCompression(silent: true);

        switch (p.action)
        {
            case 0:  _anim = StartCoroutine(AnimAssess());        break;
            case 1:  _anim = StartCoroutine(AnimCall());          break;
            case 2:  _anim = StartCoroutine(AnimOpenAirway());    break;
            case 3:  _anim = StartCoroutine(AnimCheckBreath());   break;
            case 4:  BeginCompression();  break;
            case 5:  _anim = StartCoroutine(AnimRescueBreaths()); break;
            case 6:  _anim = StartCoroutine(AnimDefibrillate());  break;
            case 7:  _anim = StartCoroutine(AnimPulseCheck());    break;
            case 8:  _anim = StartCoroutine(AnimRecovery());      break;
            case 9:  _anim = StartCoroutine(AnimReposition());    break;
            case 10: _anim = StartCoroutine(AnimTiltHead());      break;
            default: _anim = StartCoroutine(AnimIdle());          break;
        }
    }

    void HandleEpisodeEnd(StatePacket _) { EndCompression(silent:false); StartCoroutine(AnimIdle()); }
    void HandlePhaseChange(StatePacket _) { EndCompression(silent:false); SetKneeling(); }

    // ── COMPRESSION ───────────────────────────────────────────────────────────
    void BeginCompression()
    {
        _compressing = true;
        _ikActive    = false;
        _compPhase   = 0f;
        // Wrists and hands straight — locked CPR posture
        Safe(() => { _body.leftWrist.localRotation  = Quaternion.identity; });
        Safe(() => { _body.rightWrist.localRotation = Quaternion.identity; });
        Safe(() => { _body.rightHand.localEulerAngles = new Vector3(-6f,0f,0f); }); // stack
    }

    void EndCompression(bool silent)
    {
        _compressing = false;
        _ikActive    = false;
        Safe(() => { _body.rightHand.localRotation = Quaternion.identity; });
        if (!silent)
            Safe(() => { _body.spine.localEulerAngles = new Vector3(15f,0f,0f); });
    }

    void DoCompressionIK()
    {
        // 100 bpm pump cycle
        _compPhase += Time.deltaTime * (100f/60f) * Mathf.PI * 2f;
        float pump = Mathf.Abs(Mathf.Sin(_compPhase));  // 0=top 1=bottom

        // Weight-transfer lean: 20° idle → 42° at full depth
        Safe(() => {
            _body.spine.localEulerAngles = new Vector3(Mathf.Lerp(20f,42f,pump), 0f, 0f);
        });

        // Target = patient chest world pos + downward pump offset
        Vector3 chest  = GetChestPos();
        Vector3 target = chest - transform.up * (pump * 0.065f);

        // BOTH arms drive toward the same target (interlaced hands)
        SolveArms(target, target);
    }

    // ── IK ENGINE — both arms solved every frame when active ─────────────────
    void SolveArms(Vector3 leftWorld, Vector3 rightWorld)
    {
        SolveTwoBone(_body.leftUpperArm,  _body.leftForearm,  _body.leftWrist,
                     leftWorld,  left: true);
        SolveTwoBone(_body.rightUpperArm, _body.rightForearm, _body.rightWrist,
                     rightWorld, left: false);
    }

    void SolveTwoBone(Transform upper, Transform mid, Transform end,
                      Vector3 targetWorld, bool left)
    {
        if (upper == null || mid == null || end == null) return;

        float uLen = Vector3.Distance(upper.position, mid.position);
        float mLen = Vector3.Distance(mid.position,   end.position);
        float total = uLen + mLen;

        Vector3 toTarget = targetWorld - upper.position;
        float   dist     = Mathf.Clamp(toTarget.magnitude, 0.01f, total * 0.998f);

        // ── Upper arm: aim toward target ─────────────────────────────────────
        if (toTarget.sqrMagnitude > 0.0001f)
        {
            // Elbow hint — keeps elbow bending naturally inward and downward
            Vector3 hint = left
                ? new Vector3(-0.6f, -1.0f, 0.3f).normalized
                : new Vector3( 0.6f, -1.0f, 0.3f).normalized;
            Vector3 fwd  = toTarget.normalized;
            Vector3 up   = Vector3.Cross(fwd, hint).normalized;
            if (up.sqrMagnitude < 0.001f) up = Vector3.up;

            Quaternion aim = Quaternion.LookRotation(fwd, up);
            // SmoothDamp rotation for silky shoulder movement
            upper.rotation = SmoothDampQ(upper.rotation, aim,
                                          ref (left ? ref _luVel : ref _ruVel),
                                          1f / ikSpeed);
        }

        // ── Forearm: law-of-cosines bend ─────────────────────────────────────
        float cosA  = (uLen*uLen + dist*dist - mLen*mLen) / (2f*uLen*dist + 1e-5f);
        float angle = Mathf.Acos(Mathf.Clamp(cosA, -1f, 1f)) * Mathf.Rad2Deg;

        Vector3 elbowAxis = left ? Vector3.forward : Vector3.back;
        Quaternion elbowTarget = Quaternion.AngleAxis(angle, elbowAxis);
        mid.localRotation = SmoothDampQ(mid.localRotation, elbowTarget,
                                         ref (left ? ref _lfVel : ref _rfVel),
                                         1.2f / ikSpeed);   // slightly slower = smoother elbow
    }

    // ── Drift back to rest when IK is off ────────────────────────────────────
    void DriftToRest()
    {
        float smooth = restTime;
        if (_body.leftUpperArm  != null) _body.leftUpperArm.localRotation  = SmoothDampQ(_body.leftUpperArm.localRotation,  _luRest, ref _luVel, smooth);
        if (_body.rightUpperArm != null) _body.rightUpperArm.localRotation = SmoothDampQ(_body.rightUpperArm.localRotation, _ruRest, ref _ruVel, smooth);
        if (_body.leftForearm   != null) _body.leftForearm.localRotation   = SmoothDampQ(_body.leftForearm.localRotation,   _lfRest, ref _lfVel, smooth);
        if (_body.rightForearm  != null) _body.rightForearm.localRotation  = SmoothDampQ(_body.rightForearm.localRotation,  _rfRest, ref _rfVel, smooth);
        if (_body.leftWrist     != null) _body.leftWrist.localRotation     = SmoothDampQ(_body.leftWrist.localRotation,     _lwRest, ref _lwVel, smooth);
        if (_body.rightWrist    != null) _body.rightWrist.localRotation    = SmoothDampQ(_body.rightWrist.localRotation,    _rwRest, ref _rwVel, smooth);
    }

    // ── SmoothDamp for Quaternion (Unity doesn't have one built-in) ───────────
    static Quaternion SmoothDampQ(Quaternion current, Quaternion target,
                                   ref Quaternion velocity, float smoothTime)
    {
        // Convert to 4D vector space, SmoothDamp, convert back
        Vector4 c = new Vector4(current.x, current.y, current.z, current.w);
        Vector4 t = new Vector4(target.x,  target.y,  target.z,  target.w);
        if (Vector4.Dot(c, t) < 0f) t = -t;   // ensure shortest path
        Vector4 v = new Vector4(velocity.x, velocity.y, velocity.z, velocity.w);

        Vector4 r = new Vector4(
            Mathf.SmoothDamp(c.x, t.x, ref v.x, smoothTime),
            Mathf.SmoothDamp(c.y, t.y, ref v.y, smoothTime),
            Mathf.SmoothDamp(c.z, t.z, ref v.z, smoothTime),
            Mathf.SmoothDamp(c.w, t.w, ref v.w, smoothTime)
        );

        velocity = new Quaternion(v.x, v.y, v.z, v.w);
        r.Normalize();
        return new Quaternion(r.x, r.y, r.z, r.w);
    }

    // ── Action animations — BOTH arms get targets in every animation ──────────
    IEnumerator AnimAssess()
    {
        _ikActive = true;
        Vector3 ls = GetPatientShoulder(true);
        Vector3 rs = GetPatientShoulder(false);
        _leftTgt = ls + Vector3.up*0.04f;
        _rightTgt = rs + Vector3.up*0.04f;
        yield return new WaitForSeconds(0.5f);
        // Two shoulder taps
        for (int i = 0; i < 2; i++)
        {
            _leftTgt = ls + Vector3.up*0.09f;
            _rightTgt = rs + Vector3.up*0.09f;
            yield return new WaitForSeconds(0.14f);
            _leftTgt = ls; _rightTgt = rs;
            yield return new WaitForSeconds(0.14f);
        }
        yield return new WaitForSeconds(0.4f);
        _ikActive = false;
    }

    IEnumerator AnimCall()
    {
        _ikActive = true;
        // Right hand raised to ear; LEFT hand rests on knee (not dangling)
        _rightTgt = transform.position + transform.up*1.52f - transform.right*0.18f;
        _leftTgt  = transform.position + transform.up*0.55f + transform.forward*0.20f;
        Safe(() => { _body.neck.localEulerAngles = new Vector3(0f,-16f,0f); });
        yield return new WaitForSeconds(2.0f);
        Safe(() => { _body.neck.localEulerAngles = Vector3.zero; });
        _ikActive = false;
    }

    IEnumerator AnimOpenAirway()
    {
        _ikActive = true;
        Vector3 hp = GetPatientHead();
        // Both hands cup the head
        _leftTgt  = hp + transform.forward*0.11f + Vector3.up*0.02f;
        _rightTgt = hp - transform.forward*0.11f + Vector3.up*0.02f;
        Safe(() => { _body.spine.localEulerAngles = new Vector3(30f,0f,0f); });
        yield return new WaitForSeconds(1.6f);
        Safe(() => { _body.spine.localEulerAngles = new Vector3(15f,0f,0f); });
        _ikActive = false;
    }

    IEnumerator AnimCheckBreath()
    {
        // Lean cheek to patient mouth — left hand on patient chest to feel rise
        _ikActive = true;
        _leftTgt  = GetChestPos() + Vector3.up*0.04f;
        _rightTgt = GetChestPos() + Vector3.up*0.04f;
        Safe(() => { _body.spine.localEulerAngles = new Vector3(44f,-7f,0f); });
        Safe(() => { _body.neck.localEulerAngles  = new Vector3(18f, 7f,0f); });
        yield return new WaitForSeconds(1.8f);
        Safe(() => { _body.spine.localEulerAngles = new Vector3(15f,0f,0f); });
        Safe(() => { _body.neck.localEulerAngles  = Vector3.zero; });
        _ikActive = false;
    }

    IEnumerator AnimRescueBreaths()
    {
        _ikActive = true;
        // Left hand pinches nose; right hand lifts chin
        Vector3 hp = GetPatientHead();
        _leftTgt  = hp + transform.forward*0.06f + Vector3.up*0.03f;
        _rightTgt = hp - transform.forward*0.08f - Vector3.up*0.02f;
        Safe(() => { _body.spine.localEulerAngles = new Vector3(38f,-4f,0f); });
        Safe(() => { _body.neck.localEulerAngles  = new Vector3(14f, 4f,0f); });
        for (int i = 0; i < 2; i++)
        {
            float t = 0f;
            while (t < 1f) {
                t += Time.deltaTime*2.2f;
                Safe(() => { _body.spine.localEulerAngles = Vector3.Lerp(
                    new Vector3(38f,-4f,0f), new Vector3(52f,-4f,0f), t); });
                yield return null;
            }
            yield return new WaitForSeconds(0.8f);
            t = 0f;
            while (t < 1f) {
                t += Time.deltaTime*2.2f;
                Safe(() => { _body.spine.localEulerAngles = Vector3.Lerp(
                    new Vector3(52f,-4f,0f), new Vector3(38f,-4f,0f), t); });
                yield return null;
            }
            yield return new WaitForSeconds(0.3f);
        }
        Safe(() => { _body.spine.localEulerAngles = new Vector3(15f,0f,0f); });
        Safe(() => { _body.neck.localEulerAngles  = Vector3.zero; });
        _ikActive = false;
    }

    IEnumerator AnimDefibrillate()
    {
        Vector3 start = transform.position;
        Vector3 aed   = start - transform.right * 1.1f;
        yield return MoveRoot(start, aed, 0.9f);
        _ikActive = true;
        // Both hands reach toward AED
        _rightTgt = aed + Vector3.up*0.5f + transform.forward*0.1f;
        _leftTgt  = aed + Vector3.up*0.45f - transform.forward*0.08f;
        yield return new WaitForSeconds(0.7f);
        yield return MoveRoot(aed, start, 0.9f);
        // Place pads — left on left chest, right on right
        Vector3 ct = GetChestPos();
        _leftTgt  = ct + transform.forward*0.10f;
        _rightTgt = ct - transform.forward*0.10f;
        yield return new WaitForSeconds(0.5f);
        // Stand clear — both arms wide
        _leftTgt  = transform.position + transform.up*1.0f + transform.right* 0.5f;
        _rightTgt = transform.position + transform.up*1.0f - transform.right* 0.5f;
        yield return new WaitForSeconds(0.7f);
        _ikActive = false;
    }

    IEnumerator AnimPulseCheck()
    {
        _ikActive = true;
        // RIGHT hand: two fingers to patient's carotid (side of neck)
        // LEFT hand: rests on patient's chest to feel for breathing
        Vector3 neck  = GetPatientNeck();
        Vector3 chest = GetChestPos();
        _rightTgt = neck;
        _leftTgt  = chest + Vector3.up*0.03f;
        Safe(() => { _body.spine.localEulerAngles = new Vector3(26f,0f,0f); });
        yield return new WaitForSeconds(1.8f);
        Safe(() => { _body.spine.localEulerAngles = new Vector3(15f,0f,0f); });
        _ikActive = false;
    }

    IEnumerator AnimReposition()
    {
        _ikActive = true;
        Vector3 ct = GetChestPos();
        // Both hands approach chest from above
        _leftTgt  = ct + Vector3.up*0.16f;
        _rightTgt = ct + Vector3.up*0.16f;
        yield return new WaitForSeconds(0.25f);
        _leftTgt = ct; _rightTgt = ct;
        yield return new WaitForSeconds(0.4f);
        _ikActive = false;
    }

    IEnumerator AnimTiltHead()
    {
        _ikActive = true;
        Vector3 hp = GetPatientHead();
        _leftTgt  = hp + transform.forward*0.10f;
        _rightTgt = hp - transform.forward*0.10f + Vector3.down*0.03f;
        Safe(() => { _body.spine.localEulerAngles = new Vector3(32f,0f,0f); });
        yield return new WaitForSeconds(1.2f);
        Safe(() => { _body.spine.localEulerAngles = new Vector3(15f,0f,0f); });
        _ikActive = false;
    }

    IEnumerator AnimRecovery()
    {
        _ikActive = true;
        // Left on shoulder, right on hip — classic recovery position roll
        Vector3 shoulder = GetPatientShoulder(false);
        Vector3 hip      = patient != null
            ? patient.transform.position + new Vector3(-0.10f, 0.52f, -0.10f)
            : GetChestPos() + Vector3.back*0.2f;
        _leftTgt  = shoulder;
        _rightTgt = hip;
        Safe(() => { _body.spine.localEulerAngles = new Vector3(26f,10f,0f); });
        yield return new WaitForSeconds(2.0f);
        Safe(() => { _body.spine.localEulerAngles = new Vector3(15f,0f,0f); });
        _ikActive = false;
    }

    IEnumerator AnimIdle()
    {
        _ikActive = false;
        float t = 0f;
        while (true)
        {
            t += Time.deltaTime;
            Safe(() => { _body.spine.localEulerAngles =
                new Vector3(15f + Mathf.Sin(t*0.65f)*1.1f, 0f, 0f); });
            yield return null;
        }
    }

    // ── Kneeling pose — both knees on floor ───────────────────────────────────
    void SetKneeling()
    {
        if (_body == null) return;
        // Hips at ~0.48m off ground
        Safe(() => { _body.hips.localPosition = new Vector3(0f, 0.48f, 0f); });
        // Thighs: X = -90° means they point straight DOWN (proper kneeling)
        Safe(() => { _body.leftThigh.localEulerAngles  = new Vector3(-90f, 0f,  5f); });
        Safe(() => { _body.rightThigh.localEulerAngles = new Vector3(-90f, 0f, -5f); });
        // Shins: fold back horizontally (+90° = pointing backward)
        Safe(() => { _body.leftShin.localEulerAngles   = new Vector3( 90f, 0f, 0f); });
        Safe(() => { _body.rightShin.localEulerAngles  = new Vector3( 90f, 0f, 0f); });
        // Feet: slight dorsiflexion so toes contact floor
        Safe(() => { _body.leftFoot.localEulerAngles   = new Vector3(-25f, 0f, 0f); });
        Safe(() => { _body.rightFoot.localEulerAngles  = new Vector3(-25f, 0f, 0f); });
        // Torso: slight forward lean
        Safe(() => { _body.spine.localEulerAngles      = new Vector3( 15f, 0f, 0f); });
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

    // ── World-space helpers ───────────────────────────────────────────────────
    Vector3 GetChestPos()
    {
        if (patient != null) return patient.ChestWorldPosition;
        return transform.position + transform.forward*0.55f + Vector3.up*0.05f;
    }

    Vector3 GetPatientHead()
    {
        if (patient != null)
            return patient.transform.position + new Vector3(0f, 0.22f, 0.80f);
        return transform.position + transform.forward*0.85f + Vector3.up*0.20f;
    }

    // Carotid neck position — slightly below head, to the side
    Vector3 GetPatientNeck()
    {
        if (patient != null)
            return patient.transform.position + new Vector3(0.12f, 0.18f, 0.62f);
        return transform.position + transform.forward*0.70f + Vector3.up*0.18f;
    }

    Vector3 GetPatientShoulder(bool left)
    {
        float x = left ? 0.20f : -0.20f;
        if (patient != null)
            return patient.transform.position + new Vector3(x, 0.22f, 0.26f);
        return transform.position + transform.forward*0.5f + Vector3.up*0.20f;
    }

    void AddHiVis()
    {
        var s = GameObject.CreatePrimitive(PrimitiveType.Cube);
        s.name = "HiVisStripe";
        s.transform.SetParent(_body.chest, false);
        s.transform.localScale    = new Vector3(0.30f, 0.03f, 0.18f);
        s.transform.localPosition = new Vector3(0f, 0.05f, 0.08f);
        var r = s.GetComponent<Renderer>();
        var m = new Material(Shader.Find("Standard"));
        m.color = new Color(1f,0.85f,0f);
        m.SetFloat("_Glossiness",0.4f);
        r.material = m;
        var col = s.GetComponent<Collider>();
        if (col) Object.Destroy(col);
    }

    static void Safe(System.Action a) { try { a(); } catch { } }
}