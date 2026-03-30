// HumanoidRescuer.cs
// ====================
// Fully animated humanoid rescuer performing CPR on the patient.
//
// Animation set:
//   Idle kneeling → Assess (shoulder tap) → Call 911 (phone hand raise)
//   → Head tilt (hands cup patient's head) → Check breathing (lean in)
//   → Reposition hands (adjust to lower sternum) → Compression cycle
//     (weight-shift forward, both arms locked, 30× pumps at 100bpm)
//   → Rescue breaths (lean to face, mouth approach)
//   → Defibrillate (stand, walk to AED, place pads, stand clear, shock)
//   → Pulse check (two fingers to neck)
//   → Recovery position (roll patient, lean back)
//
// Hand interception with patient chest:
//   Both wrists are driven by a manual IK solver toward the patient's
//   ChestWorldPosition. When the hands arrive, the chest compression
//   spring in HumanoidPatient activates — giving the appearance of
//   genuine physical contact without a physics engine.
//
// No Animation Rigging package required — pure transform-based IK.

using System.Collections;
using UnityEngine;

public class HumanoidRescuer : MonoBehaviour
{
    [Header("References")]
    [Tooltip("Assign the Patient GameObject so the rescuer knows where to reach")]
    public HumanoidPatient patient;

    [Header("Appearance")]
    public Color skinColor    = new Color(0.87f, 0.72f, 0.56f);
    public Color shirtColor   = new Color(0.22f, 0.42f, 0.74f);  // paramedic blue
    public Color trouserColor = new Color(0.20f, 0.20f, 0.25f);

    [Header("IK Settings")]
    public float ikSpeed      = 8f;    // how fast hands move to target
    public float ikRestDamp   = 4f;    // how fast hands return to rest

    // ── Runtime ──────────────────────────────────────────────────────────────
    private HumanoidBuilder.Body _body;
    private Vector3 _leftHandTarget;
    private Vector3 _rightHandTarget;
    // (hand rest is now driven by arm rotations, not cached positions)
    private bool    _ikActive         = false;
    private bool    _compressing      = false;
    private float   _compressionPhase = 0f;
    private Coroutine _currentAnim;

    // Expose hand world pos for scene manager (e.g. particle FX on contact)
    public Vector3 LeftHandWorld  => _body?.leftHand.position  ?? transform.position;
    public Vector3 RightHandWorld => _body?.rightHand.position ?? transform.position;

    void Awake()
    {
        // Rescuer starts kneeling beside patient — positioned in CPRSceneEnvironment
        _body = HumanoidBuilder.Build(
            transform, skinColor, shirtColor, trouserColor, 1f);

        // Add blue paramedic vest stripe
        AddVestStripe();

        // Kneeling pose: hips lowered, thighs rotated forward
        SetKneeling();

        // Set initial arm rest rotations
        _body.leftUpperArm.localRotation  = Quaternion.Euler(15f, 0f, -20f);
        _body.rightUpperArm.localRotation = Quaternion.Euler(15f, 0f,  20f);

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
        // Compression pump cycle (driven every frame when active)
        if (_compressing)
        {
            _compressionPhase += Time.deltaTime * (100f / 60f) * Mathf.PI * 2f;
            float pump = Mathf.Abs(Mathf.Sin(_compressionPhase)) * 0.12f;

            // Lean torso forward on downstroke — body weight transfer
            _body.spine.localEulerAngles = new Vector3(
                Mathf.Lerp(15f, 35f, pump / 0.12f), 0, 0);

            // Drive both hands toward chest target with depth modulation
            Vector3 chestPos = GetChestTarget();
            Vector3 pumpOffset = transform.up * (-pump);
            MoveHandsToward(chestPos + pumpOffset, chestPos + pumpOffset, ikSpeed);
        }
        else
        {
            // IK toward target if active, else drift back to rest
            if (_ikActive)
                MoveHandsToward(_leftHandTarget, _rightHandTarget, ikSpeed);
        }
    }

    // ── Event handlers ────────────────────────────────────────────────────────
    void HandleState(StatePacket p)
    {
        if (_currentAnim != null) StopCoroutine(_currentAnim);

        switch (p.action)
        {
            case 0:  _currentAnim = StartCoroutine(AnimAssess());           break;
            case 1:  _currentAnim = StartCoroutine(AnimCallEmergency());    break;
            case 2:  _currentAnim = StartCoroutine(AnimOpenAirway());       break;
            case 3:  _currentAnim = StartCoroutine(AnimCheckBreathing());   break;
            case 4:  BeginCompression(); break;
            case 5:  _currentAnim = StartCoroutine(AnimRescueBreaths());    break;
            case 6:  _currentAnim = StartCoroutine(AnimDefibrillate());     break;
            case 7:  _currentAnim = StartCoroutine(AnimPulseCheck());       break;
            case 8:  _currentAnim = StartCoroutine(AnimRecoveryPosition()); break;
            case 9:  _currentAnim = StartCoroutine(AnimRepositionHands());  break;
            case 10: _currentAnim = StartCoroutine(AnimTiltHead());         break;
            case 11: EndCompression(); _currentAnim = StartCoroutine(AnimIdle()); break;
        }
    }

    void HandleEpisodeEnd(StatePacket p)
    {
        EndCompression();
        if (_currentAnim != null) StopCoroutine(_currentAnim);
        StartCoroutine(AnimIdle());
    }

    void HandlePhaseChange(StatePacket p)
    {
        EndCompression();
        if (_currentAnim != null) StopCoroutine(_currentAnim);
        SetKneeling();
    }

    // ── Action animations ─────────────────────────────────────────────────────

    IEnumerator AnimAssess()
    {
        // Reach forward, tap patient's shoulders
        _ikActive = true;
        Vector3 leftShoulder  = GetPatientShoulder(left: true);
        Vector3 rightShoulder = GetPatientShoulder(left: false);
        _leftHandTarget  = leftShoulder;
        _rightHandTarget = rightShoulder;
        yield return new WaitForSeconds(0.6f);

        // Two taps
        for (int i = 0; i < 2; i++)
        {
            _leftHandTarget  = leftShoulder + transform.up * 0.06f;
            _rightHandTarget = rightShoulder + transform.up * 0.06f;
            yield return new WaitForSeconds(0.15f);
            _leftHandTarget  = leftShoulder;
            _rightHandTarget = rightShoulder;
            yield return new WaitForSeconds(0.15f);
        }
        yield return new WaitForSeconds(0.4f);
        yield return ReturnToRest();
    }

    IEnumerator AnimCallEmergency()
    {
        // Right hand raises to ear — phone gesture
        _ikActive = true;
        _rightHandTarget = transform.position + transform.up * 1.6f
                           + transform.right * (-0.2f)
                           + transform.forward * 0.1f;
        _leftHandTarget  = GetRestWorldLeft();
        yield return new WaitForSeconds(1.5f);

        // Head turns slightly (simulated — rotate neck)
        _body.neck.localEulerAngles = new Vector3(0, -20f, 0);
        yield return new WaitForSeconds(1.0f);
        _body.neck.localEulerAngles = Vector3.zero;
        yield return ReturnToRest();
    }

    IEnumerator AnimOpenAirway()
    {
        // Both hands cup patient's head — head-tilt chin-lift
        _ikActive = true;
        Vector3 headPos = GetPatientHeadPos();
        _leftHandTarget  = headPos + transform.right * 0.12f;
        _rightHandTarget = headPos - transform.right * 0.12f;

        // Lean torso forward
        _body.spine.localEulerAngles = new Vector3(25f, 0, 0);
        yield return new WaitForSeconds(1.2f);
        _body.spine.localEulerAngles = new Vector3(15f, 0, 0);
        yield return ReturnToRest();
    }

    IEnumerator AnimCheckBreathing()
    {
        // Lean cheek near patient's mouth — look, listen, feel
        _body.spine.localEulerAngles = new Vector3(40f, -10f, 0);
        _body.neck.localEulerAngles  = new Vector3(20f, 10f, 0);
        yield return new WaitForSeconds(1.5f);
        _body.spine.localEulerAngles = new Vector3(15f, 0, 0);
        _body.neck.localEulerAngles  = Vector3.zero;
    }

    void BeginCompression()
    {
        _compressing      = true;
        _compressionPhase = 0f;
        _ikActive         = false;
        // Lock elbows straight — arms rigid for CPR
        _body.leftForearm.localEulerAngles  = Vector3.zero;
        _body.rightForearm.localEulerAngles = Vector3.zero;
        // Interlace hands — slightly rotate right hand onto left (no position change)
        _body.rightHand.localRotation = Quaternion.Euler(-10f, 0f, 0f);
    }

    void EndCompression()
    {
        _compressing = false;
        _ikActive    = false;
        _body.spine.localEulerAngles = new Vector3(15f, 0, 0);
        // Restore hand rotation
        _body.rightHand.localRotation = Quaternion.identity;
    }

    IEnumerator AnimRescueBreaths()
    {
        EndCompression();
        _body.spine.localEulerAngles = new Vector3(35f, -5f, 0);
        _body.neck.localEulerAngles  = new Vector3(15f,  5f, 0);

        // Move face close to patient's mouth
        for (int breath = 0; breath < 2; breath++)
        {
            // Lean in
            float t = 0f;
            while (t < 1f) {
                t += Time.deltaTime * 2f;
                _body.spine.localEulerAngles =
                    Vector3.Lerp(new Vector3(35f,-5f,0), new Vector3(50f,-5f,0), t);
                yield return null;
            }
            yield return new WaitForSeconds(1.0f);  // breath duration
            // Lean back
            t = 0f;
            while (t < 1f) {
                t += Time.deltaTime * 2f;
                _body.spine.localEulerAngles =
                    Vector3.Lerp(new Vector3(50f,-5f,0), new Vector3(35f,-5f,0), t);
                yield return null;
            }
            yield return new WaitForSeconds(0.4f);
        }
        _body.spine.localEulerAngles = new Vector3(15f, 0, 0);
        _body.neck.localEulerAngles  = Vector3.zero;
    }

    IEnumerator AnimDefibrillate()
    {
        EndCompression();
        // Stand up briefly
        yield return LerpSpine(new Vector3(15,0,0), new Vector3(-5,0,0), 0.5f);

        // Step toward AED (translate root slightly)
        Vector3 startPos = transform.position;
        Vector3 aedPos   = startPos + transform.right * 1.1f;
        yield return MoveRoot(startPos, aedPos, 0.8f);

        // Reach for AED
        _ikActive = true;
        _rightHandTarget = aedPos + Vector3.up * 0.5f;
        yield return new WaitForSeconds(0.6f);

        // Return with pads
        yield return MoveRoot(aedPos, startPos, 0.8f);

        // Place pads — both hands to patient chest
        Vector3 ct = GetChestTarget();
        _leftHandTarget  = ct + Vector3.forward * 0.08f;
        _rightHandTarget = ct - Vector3.forward * 0.08f;
        yield return new WaitForSeconds(0.5f);

        // STAND CLEAR — arms wide
        _leftHandTarget  = transform.position + transform.right *  0.6f + transform.up * 1.0f;
        _rightHandTarget = transform.position + transform.right * -0.6f + transform.up * 1.0f;
        yield return new WaitForSeconds(0.8f);

        // Shock delivered
        yield return ReturnToRest();
        yield return LerpSpine(new Vector3(-5,0,0), new Vector3(15,0,0), 0.4f);
    }

    IEnumerator AnimPulseCheck()
    {
        // Two fingers to carotid (side of neck)
        _ikActive = true;
        _rightHandTarget = GetPatientHeadPos() - transform.up * 0.05f
                           + transform.right  * 0.08f;
        _leftHandTarget  = GetRestWorldLeft();
        yield return new WaitForSeconds(1.2f);
        yield return ReturnToRest();
    }

    IEnumerator AnimRepositionHands()
    {
        // Hands lift and reposition onto lower sternum
        _ikActive = true;
        _leftHandTarget  = GetChestTarget() + transform.up *  0.15f;
        _rightHandTarget = GetChestTarget() + transform.up *  0.15f;
        yield return new WaitForSeconds(0.3f);
        _leftHandTarget  = GetChestTarget();
        _rightHandTarget = GetChestTarget();
        yield return new WaitForSeconds(0.4f);
        yield return ReturnToRest();
    }

    IEnumerator AnimTiltHead()
    {
        _ikActive = true;
        Vector3 headPos = GetPatientHeadPos();
        _leftHandTarget  = headPos + Vector3.forward * 0.08f;
        _rightHandTarget = headPos + Vector3.back    * 0.05f;
        _body.spine.localEulerAngles = new Vector3(30f, 0, 0);
        yield return new WaitForSeconds(1.0f);
        _body.spine.localEulerAngles = new Vector3(15f, 0, 0);
        yield return ReturnToRest();
    }

    IEnumerator AnimRecoveryPosition()
    {
        // Push patient onto side — hands on shoulder and hip
        _ikActive = true;
        _leftHandTarget  = GetPatientShoulder(left: false);
        _rightHandTarget = patient != null
            ? patient.transform.position + Vector3.up * 0.5f + Vector3.right * 0.15f
            : GetChestTarget();
        _body.spine.localEulerAngles = new Vector3(25f, 10f, 0);
        yield return new WaitForSeconds(0.5f);

        // Sustained push — 1.5 seconds
        yield return new WaitForSeconds(1.5f);
        _body.spine.localEulerAngles = new Vector3(15f, 0, 0);
        yield return ReturnToRest();
    }

    IEnumerator AnimIdle()
    {
        yield return ReturnToRest();
        // Gentle idle sway
        float t = 0f;
        while (true)
        {
            t += Time.deltaTime;
            _body.spine.localEulerAngles =
                new Vector3(15f + Mathf.Sin(t * 0.8f) * 1.5f, 0, 0);
            yield return null;
        }
    }

    // ── IK helpers ────────────────────────────────────────────────────────────

    // ── Two-bone IK — rotation only, hands NEVER repositioned directly ────────
    // Driving hand.position directly detaches it from the arm chain.
    // Instead: rotate upper arm to aim at target, use law of cosines for
    // forearm bend. The hand follows naturally as the last link in the chain.
    void MoveHandsToward(Vector3 leftWorld, Vector3 rightWorld, float speed)
    {
        SolveTwoBoneIK(_body.leftUpperArm,  _body.leftForearm,  _body.leftHand,
                       leftWorld,  left: true);
        SolveTwoBoneIK(_body.rightUpperArm, _body.rightForearm, _body.rightHand,
                       rightWorld, left: false);
    }

    void SolveTwoBoneIK(Transform upper, Transform forearm, Transform hand,
                        Vector3 targetWorld, bool left)
    {
        float upperLen   = Vector3.Distance(upper.position, forearm.position);
        float forearmLen = Vector3.Distance(forearm.position, hand.position);
        float totalLen   = upperLen + forearmLen;

        Vector3 toTarget = targetWorld - upper.position;
        float   dist     = Mathf.Clamp(toTarget.magnitude, 0.01f, totalLen * 0.999f);

        // ── Step 1: rotate upper arm to face target ───────────────────────
        if (toTarget.sqrMagnitude > 0.0001f)
        {
            // Hint vector keeps the elbow bending naturally (downward / outward)
            Vector3 hint     = left ? -Vector3.right + Vector3.down
                                    :  Vector3.right + Vector3.down;
            Vector3 fwd      = toTarget.normalized;
            Vector3 up       = Vector3.Cross(fwd, hint).normalized;
            if (up.sqrMagnitude < 0.001f) up = Vector3.up;

            Quaternion target = Quaternion.LookRotation(fwd, up);
            upper.rotation    = Quaternion.Slerp(upper.rotation, target,
                                                  Time.deltaTime * ikSpeed);
        }

        // ── Step 2: bend forearm by law of cosines ────────────────────────
        float cosElbow = (upperLen * upperLen + dist * dist - forearmLen * forearmLen)
                       / (2f * upperLen * dist + 0.0001f);
        float elbowAngle = Mathf.Acos(Mathf.Clamp(cosElbow, -1f, 1f)) * Mathf.Rad2Deg;

        // Forearm bends around its own local Z (elbow axis)
        Quaternion bendTarget = Quaternion.AngleAxis(elbowAngle,
                                    left ? Vector3.forward : Vector3.back);
        forearm.localRotation = Quaternion.Slerp(forearm.localRotation, bendTarget,
                                                   Time.deltaTime * ikSpeed);
    }

    IEnumerator ReturnToRest()
    {
        _ikActive = false;
        // Restore arm rotations to kneeling rest pose — never move hand.position
        float t = 0f;
        Quaternion luStart = _body.leftUpperArm.localRotation;
        Quaternion ruStart = _body.rightUpperArm.localRotation;
        Quaternion lfStart = _body.leftForearm.localRotation;
        Quaternion rfStart = _body.rightForearm.localRotation;

        // Natural kneeling rest: arms hanging slightly forward at sides
        Quaternion luRest  = Quaternion.Euler(  15f, 0f,  -20f);
        Quaternion ruRest  = Quaternion.Euler(  15f, 0f,   20f);
        Quaternion lfRest  = Quaternion.Euler(  20f, 0f,    0f);
        Quaternion rfRest  = Quaternion.Euler(  20f, 0f,    0f);

        while (t < 1f)
        {
            t += Time.deltaTime * ikRestDamp;
            float s = Mathf.SmoothStep(0, 1, t);
            _body.leftUpperArm.localRotation  = Quaternion.Slerp(luStart, luRest, s);
            _body.rightUpperArm.localRotation = Quaternion.Slerp(ruStart, ruRest, s);
            _body.leftForearm.localRotation   = Quaternion.Slerp(lfStart, lfRest, s);
            _body.rightForearm.localRotation  = Quaternion.Slerp(rfStart, rfRest, s);
            yield return null;
        }
    }

    // ── Pose helpers ──────────────────────────────────────────────────────────

    void SetKneeling()
    {
        // Hips lowered, thighs angled forward, shins vertical
        _body.hips.localPosition = new Vector3(0, 0.55f, 0);
        _body.leftThigh.localEulerAngles  = new Vector3(-80f, 0,  8f);
        _body.rightThigh.localEulerAngles = new Vector3(-80f, 0, -8f);
        _body.leftShin.localEulerAngles   = new Vector3( 80f, 0,  0f);
        _body.rightShin.localEulerAngles  = new Vector3( 80f, 0,  0f);
        _body.spine.localEulerAngles      = new Vector3( 15f, 0,  0f);
    }

    IEnumerator LerpSpine(Vector3 from, Vector3 to, float duration)
    {
        float t = 0f;
        while (t < 1f)
        {
            t += Time.deltaTime / duration;
            _body.spine.localEulerAngles = Vector3.Lerp(from, to, Mathf.SmoothStep(0,1,t));
            yield return null;
        }
    }

    IEnumerator MoveRoot(Vector3 from, Vector3 to, float duration)
    {
        float t = 0f;
        while (t < 1f)
        {
            t += Time.deltaTime / duration;
            transform.position = Vector3.Lerp(from, to, Mathf.SmoothStep(0,1,t));
            yield return null;
        }
    }

    // ── World position helpers ────────────────────────────────────────────────
    Vector3 GetChestTarget() =>
        patient != null ? patient.ChestWorldPosition : transform.position + transform.forward * 0.5f;

    Vector3 GetPatientHeadPos() =>
        patient != null
            ? patient.transform.position + Vector3.up * 0.1f + Vector3.forward * 0.85f
            : transform.position + transform.forward * 0.8f + Vector3.up * 0.15f;

    Vector3 GetPatientShoulder(bool left) =>
        patient != null
            ? patient.transform.position + Vector3.up * 0.25f
              + (left ? Vector3.right : Vector3.left) * 0.22f
              + Vector3.forward * 0.3f
            : transform.position + transform.forward * 0.5f;

    Vector3 GetRestWorldLeft() =>
        _body.leftForearm != null
            ? _body.leftForearm.position + _body.leftForearm.forward * 0.25f
            : transform.position + Vector3.up * 0.8f;

    // ── Cosmetic extras ───────────────────────────────────────────────────────
    void AddVestStripe()
    {
        // Bright yellow hi-vis stripe across chest — paramedic identifier
        var stripe = GameObject.CreatePrimitive(PrimitiveType.Cube);
        stripe.name = "HiVisStripe";
        stripe.transform.SetParent(_body.chest, false);
        stripe.transform.localScale    = new Vector3(0.30f, 0.03f, 0.17f);
        stripe.transform.localPosition = new Vector3(0, 0.05f, 0.08f);
        var r = stripe.GetComponent<Renderer>();
        var mat = new Material(Shader.Find("Standard"));
        mat.color = new Color(1f, 0.85f, 0f);   // hi-vis yellow
        mat.SetFloat("_Glossiness", 0.4f);
        r.material = mat;
        var col = stripe.GetComponent<Collider>();
        if (col) Destroy(col);
    }
}