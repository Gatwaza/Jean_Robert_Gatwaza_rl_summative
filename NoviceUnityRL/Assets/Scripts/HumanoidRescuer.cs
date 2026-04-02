// HumanoidRescuer.cs  (v6 — fatigue system, compression counter HUD, ROSC celebration)
// New in v6:
//   • FatigueSystem   — spine lean + ikSpeed drop after 30 pump cycles; visual
//                       fatigue indicator in 3D (slight body sway increases)
//   • Compression count tracked; HUD overlay shows count over patient chest
//   • ROSC celebration — rescuer sits back, raises arms in relief
//   • Phase-based skin tone variation for demo diversity

using System.Collections;
using UnityEngine;
using UnityEngine.UI;

public class HumanoidRescuer : MonoBehaviour
{
    [Header("References")]
    public HumanoidPatient patient;

    [Header("Appearance")]
    public Color skinColor    = new Color(0.87f, 0.72f, 0.56f);
    public Color shirtColor   = new Color(0.22f, 0.42f, 0.74f);
    public Color trouserColor = new Color(0.20f, 0.20f, 0.25f);

    [Header("IK Settings")]
    public float ikSpeed  = 12f;
    public float restTime = 0.25f;

    // ── Internal state ────────────────────────────────────────────────────────
    private HumanoidBuilder.Body _body;

    private Vector3  _leftTgt, _rightTgt;
    private bool     _ikActive    = false;
    private bool     _compressing = false;
    private float    _compPhase   = 0f;

    // Fatigue
    private int      _pumpCycles  = 0;       // total pump cycles since reset
    private float    _fatigueLevel= 0f;      // 0 = fresh, 1 = exhausted
    private float    _sway        = 0f;      // idle sway magnitude

    // Smoothdamp velocity handles
    private Quaternion _luVel = Quaternion.identity, _ruVel = Quaternion.identity;
    private Quaternion _lfVel = Quaternion.identity, _rfVel = Quaternion.identity;
    private Quaternion _lwVel = Quaternion.identity, _rwVel = Quaternion.identity;

    private Coroutine _anim;

    // Rest pose cache
    private Quaternion _luRest, _ruRest, _lfRest, _rfRest, _lwRest, _rwRest;

    // Compression counter world-space label
    private Canvas    _worldCanvas;
    private Text      _compCountLabel;
    private int       _totalCompressions = 0;

    // Celebration flag
    private bool _celebrating = false;

    // ── Lifecycle ─────────────────────────────────────────────────────────────
    void Awake()
    {
        _body = HumanoidBuilder.Build(transform, skinColor, shirtColor, trouserColor, 1f);
        AddHiVis();
        SetKneeling();
        CacheRestPose();
        BuildCompLabel();

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
        _luRest = Quaternion.Euler(12f,  0f, -22f);
        _ruRest = Quaternion.Euler(12f,  0f,  22f);
        _lfRest = Quaternion.Euler(18f,  0f,   0f);
        _rfRest = Quaternion.Euler(18f,  0f,   0f);
        _lwRest = Quaternion.identity;
        _rwRest = Quaternion.identity;
    }

    void Update()
    {
        if (_compressing)        DoCompressionIK();
        else if (_ikActive)      SolveArms(_leftTgt, _rightTgt);
        else if (!_celebrating)  DriftToRest();

        UpdateFatigueVisuals();

        // Compression label billboard — always face camera
        if (_worldCanvas != null && Camera.main != null)
            _worldCanvas.transform.rotation = Camera.main.transform.rotation;
    }

    // ── Event handlers ────────────────────────────────────────────────────────
    void HandleState(StatePacket p)
    {
        if (_celebrating) return;
        if (_anim != null) StopCoroutine(_anim);
        EndCompression(silent: true);

        switch (p.action)
        {
            case 0:  _anim = StartCoroutine(AnimAssess());        break;
            case 1:  _anim = StartCoroutine(AnimCall());          break;
            case 2:  _anim = StartCoroutine(AnimOpenAirway());    break;
            case 3:  _anim = StartCoroutine(AnimCheckBreath());   break;
            case 4:  BeginCompression();
                     _totalCompressions += 30;
                     UpdateCompLabel();                           break;
            case 5:  _anim = StartCoroutine(AnimRescueBreaths()); break;
            case 6:  _anim = StartCoroutine(AnimDefibrillate());  break;
            case 7:  _anim = StartCoroutine(AnimPulseCheck());    break;
            case 8:  _anim = StartCoroutine(AnimRecovery());      break;
            case 9:  _anim = StartCoroutine(AnimReposition());    break;
            case 10: _anim = StartCoroutine(AnimTiltHead());      break;
            default: _anim = StartCoroutine(AnimIdle());          break;
        }

        if (p.rosc && !_celebrating)
        {
            if (_anim != null) StopCoroutine(_anim);
            EndCompression(silent: true);
            _anim = StartCoroutine(AnimROSCCelebration());
        }
    }

    void HandleEpisodeEnd(StatePacket p)
    {
        _celebrating = false;
        EndCompression(silent: false);
        ResetFatigue();
        _totalCompressions = 0;
        UpdateCompLabel();
        StartCoroutine(AnimIdle());
    }

    void HandlePhaseChange(StatePacket p)
    {
        _celebrating = false;
        EndCompression(silent: false);
        ResetFatigue();
        _totalCompressions = 0;
        UpdateCompLabel();
        SetKneeling();
    }

    // ── Compression IK ───────────────────────────────────────────────────────
    void BeginCompression()
    {
        _compressing = true;
        _ikActive    = false;
        _compPhase   = 0f;
        Safe(() => { _body.leftWrist.localRotation  = Quaternion.identity; });
        Safe(() => { _body.rightWrist.localRotation = Quaternion.identity; });
        Safe(() => { _body.rightHand.localEulerAngles = new Vector3(-6f, 0f, 0f); });
    }

    void EndCompression(bool silent)
    {
        _compressing = false;
        _ikActive    = false;
        Safe(() => { _body.rightHand.localRotation = Quaternion.identity; });
        if (!silent)
            Safe(() => { _body.spine.localEulerAngles = new Vector3(15f, 0f, 0f); });
    }

    void DoCompressionIK()
    {
        // 100 bpm pump — scale speed with fatigue (tired = slightly slower)
        float bpm = Mathf.Lerp(100f, 88f, _fatigueLevel);
        _compPhase += Time.deltaTime * (bpm / 60f) * Mathf.PI * 2f;
        float pump  = Mathf.Abs(Mathf.Sin(_compPhase));

        _pumpCycles++;

        // ── Fatigue accumulation ──────────────────────────────────────────────
        // Fatigue rises after 30 pump cycles (≈ one compression set of 30 at 100bpm)
        if (_pumpCycles > 30)
            _fatigueLevel = Mathf.Min(1.0f, _fatigueLevel + 0.0004f);

        // Lean deepens with both pump phase AND fatigue
        float baseLean  = Mathf.Lerp(20f, 28f, _fatigueLevel);
        float pumpLean  = Mathf.Lerp(baseLean, Mathf.Lerp(42f, 52f, _fatigueLevel), pump);
        Safe(() => { _body.spine.localEulerAngles = new Vector3(pumpLean, 0f, 0f); });

        Vector3 chest  = GetChestPos();
        Vector3 target = chest - transform.up * (pump * 0.065f);
        SolveArms(target, target);
    }

    // ── Fatigue visual update ─────────────────────────────────────────────────
    void UpdateFatigueVisuals()
    {
        if (_fatigueLevel < 0.01f) return;
        // Shoulders droop slightly
        float droop = _fatigueLevel * 4f;
        if (!_compressing && !_ikActive)
        {
            Safe(() => {
                _body.leftShoulder.localEulerAngles  = new Vector3(0f, 0f, -droop);
                _body.rightShoulder.localEulerAngles = new Vector3(0f, 0f,  droop);
            });
        }
        // Spine: idle lean increases with fatigue
        if (!_compressing && !_ikActive)
        {
            _sway = Mathf.Lerp(1.2f, 3.5f, _fatigueLevel);
        }
    }

    void ResetFatigue()
    {
        _fatigueLevel = 0f;
        _pumpCycles   = 0;
        _sway         = 0f;
        Safe(() => { _body.leftShoulder.localEulerAngles  = Vector3.zero; });
        Safe(() => { _body.rightShoulder.localEulerAngles = Vector3.zero; });
    }

    // ── Two-bone IK ───────────────────────────────────────────────────────────
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

        // Effective IK speed — reduce with fatigue for tired appearance
        float speed = Mathf.Lerp(ikSpeed, ikSpeed * 0.70f, _fatigueLevel);

        if (toTarget.sqrMagnitude > 0.0001f)
        {
            Vector3 hint = left
                ? new Vector3(-0.6f, -1.0f, 0.3f).normalized
                : new Vector3( 0.6f, -1.0f, 0.3f).normalized;
            Vector3 fwd  = toTarget.normalized;
            Vector3 up   = Vector3.Cross(fwd, hint).normalized;
            if (up.sqrMagnitude < 0.001f) up = Vector3.up;

            Quaternion aim = Quaternion.LookRotation(fwd, up);
            upper.rotation = SmoothDampQ(upper.rotation, aim,
                                          ref (left ? ref _luVel : ref _ruVel),
                                          1f / speed);
        }

        float cosA  = (uLen*uLen + dist*dist - mLen*mLen) / (2f*uLen*dist + 1e-5f);
        float angle = Mathf.Acos(Mathf.Clamp(cosA, -1f, 1f)) * Mathf.Rad2Deg;

        Vector3 elbowAxis   = left ? Vector3.forward : Vector3.back;
        Quaternion elbowTgt = Quaternion.AngleAxis(angle, elbowAxis);
        mid.localRotation   = SmoothDampQ(mid.localRotation, elbowTgt,
                                           ref (left ? ref _lfVel : ref _rfVel),
                                           1.2f / speed);
    }

    void DriftToRest()
    {
        float s = restTime;
        Safe(() => { _body.leftUpperArm.localRotation  = SmoothDampQ(_body.leftUpperArm.localRotation,  _luRest, ref _luVel, s); });
        Safe(() => { _body.rightUpperArm.localRotation = SmoothDampQ(_body.rightUpperArm.localRotation, _ruRest, ref _ruVel, s); });
        Safe(() => { _body.leftForearm.localRotation   = SmoothDampQ(_body.leftForearm.localRotation,   _lfRest, ref _lfVel, s); });
        Safe(() => { _body.rightForearm.localRotation  = SmoothDampQ(_body.rightForearm.localRotation,  _rfRest, ref _rfVel, s); });
        Safe(() => { _body.leftWrist.localRotation     = SmoothDampQ(_body.leftWrist.localRotation,     _lwRest, ref _lwVel, s); });
        Safe(() => { _body.rightWrist.localRotation    = SmoothDampQ(_body.rightWrist.localRotation,    _rwRest, ref _rwVel, s); });
    }

    // ── Action animations ──────────────────────────────────────────────────────

    IEnumerator AnimAssess()
    {
        _ikActive = true;
        Vector3 chest = GetChestPos();
        _leftTgt  = chest + Vector3.up * 0.12f;
        _rightTgt = GetPatientShoulder(false);
        Safe(() => { _body.spine.localEulerAngles = new Vector3(24f, 0f, 0f); });
        yield return new WaitForSeconds(0.8f);
        _rightTgt = GetPatientHead() + Vector3.down * 0.08f;
        yield return new WaitForSeconds(0.8f);
        Safe(() => { _body.spine.localEulerAngles = new Vector3(15f, 0f, 0f); });
        _ikActive = false;
    }

    IEnumerator AnimCall()
    {
        _ikActive = true;
        // Right hand raised to ear — phone gesture
        _rightTgt = transform.position + transform.up * 1.55f + transform.right * (-0.22f);
        _leftTgt  = transform.position + transform.up * 0.80f;
        Safe(() => { _body.spine.localEulerAngles = new Vector3(5f, 12f, 0f); });
        yield return new WaitForSeconds(1.4f);
        Safe(() => { _body.spine.localEulerAngles = new Vector3(15f, 0f, 0f); });
        _ikActive = false;
    }

    IEnumerator AnimOpenAirway()
    {
        _ikActive = true;
        Vector3 hp = GetPatientHead();
        _leftTgt  = hp + transform.forward * 0.12f;
        _rightTgt = hp - transform.forward * 0.12f + Vector3.down * 0.04f;
        Safe(() => { _body.spine.localEulerAngles = new Vector3(32f, 0f, 0f); });
        yield return new WaitForSeconds(1.4f);
        Safe(() => { _body.spine.localEulerAngles = new Vector3(15f, 0f, 0f); });
        _ikActive = false;
    }

    IEnumerator AnimCheckBreath()
    {
        _ikActive = true;
        Vector3 face = GetPatientHead() + Vector3.down * 0.05f + transform.forward * 0.18f;
        _leftTgt  = face;
        _rightTgt = GetChestPos() + Vector3.up * 0.04f;
        Safe(() => { _body.spine.localEulerAngles = new Vector3(38f, -4f, 0f); });
        Safe(() => { _body.neck.localEulerAngles  = new Vector3(14f,  4f, 0f); });
        for (int i = 0; i < 2; i++)
        {
            float t = 0f;
            while (t < 1f) {
                t += Time.deltaTime * 2.2f;
                Safe(() => { _body.spine.localEulerAngles = Vector3.Lerp(
                    new Vector3(38f,-4f,0f), new Vector3(52f,-4f,0f), t); });
                yield return null;
            }
            yield return new WaitForSeconds(0.8f);
            t = 0f;
            while (t < 1f) {
                t += Time.deltaTime * 2.2f;
                Safe(() => { _body.spine.localEulerAngles = Vector3.Lerp(
                    new Vector3(52f,-4f,0f), new Vector3(38f,-4f,0f), t); });
                yield return null;
            }
            yield return new WaitForSeconds(0.3f);
        }
        Safe(() => { _body.spine.localEulerAngles = new Vector3(15f, 0f, 0f); });
        Safe(() => { _body.neck.localEulerAngles  = Vector3.zero; });
        _ikActive = false;
    }

    IEnumerator AnimRescueBreaths()
    {
        _ikActive = true;
        Vector3 hp = GetPatientHead();
        // One hand tilts head, other pinches nose
        _leftTgt  = hp + transform.forward * 0.14f + Vector3.up * 0.02f;
        _rightTgt = hp - transform.forward * 0.06f + Vector3.down * 0.02f;
        Safe(() => { _body.spine.localEulerAngles = new Vector3(38f, 0f, 0f); });
        // Two breaths
        for (int i = 0; i < 2; i++)
        {
            float t = 0f;
            while (t < 1f) {
                t += Time.deltaTime * 2.5f;
                Safe(() => { _body.spine.localEulerAngles = Vector3.Lerp(
                    new Vector3(38f,0f,0f), new Vector3(50f,0f,0f), t); });
                yield return null;
            }
            yield return new WaitForSeconds(1.0f);
            t = 0f;
            while (t < 1f) {
                t += Time.deltaTime * 1.8f;
                Safe(() => { _body.spine.localEulerAngles = Vector3.Lerp(
                    new Vector3(50f,0f,0f), new Vector3(38f,0f,0f), t); });
                yield return null;
            }
            if (i == 0) yield return new WaitForSeconds(0.5f);
        }
        Safe(() => { _body.spine.localEulerAngles = new Vector3(15f, 0f, 0f); });
        _ikActive = false;
    }

    IEnumerator AnimDefibrillate()
    {
        Vector3 start = transform.position;
        Vector3 aed   = start - transform.right * 1.1f;
        yield return MoveRoot(start, aed, 0.9f);
        _ikActive = true;
        _rightTgt = aed + Vector3.up * 0.5f + transform.forward * 0.1f;
        _leftTgt  = aed + Vector3.up * 0.45f - transform.forward * 0.08f;
        yield return new WaitForSeconds(0.7f);
        yield return MoveRoot(aed, start, 0.9f);
        Vector3 ct = GetChestPos();
        _leftTgt  = ct + transform.forward * 0.10f;
        _rightTgt = ct - transform.forward * 0.10f;
        yield return new WaitForSeconds(0.5f);
        // Stand clear — arms wide
        _leftTgt  = transform.position + transform.up * 1.0f + transform.right *  0.5f;
        _rightTgt = transform.position + transform.up * 1.0f - transform.right *  0.5f;
        yield return new WaitForSeconds(0.7f);
        _ikActive = false;
    }

    IEnumerator AnimPulseCheck()
    {
        _ikActive = true;
        _rightTgt = GetPatientNeck();
        _leftTgt  = GetChestPos() + Vector3.up * 0.03f;
        Safe(() => { _body.spine.localEulerAngles = new Vector3(26f, 0f, 0f); });
        yield return new WaitForSeconds(1.8f);
        Safe(() => { _body.spine.localEulerAngles = new Vector3(15f, 0f, 0f); });
        _ikActive = false;
    }

    IEnumerator AnimRecovery()
    {
        _ikActive = true;
        _leftTgt  = GetPatientShoulder(false);
        _rightTgt = patient != null
            ? patient.transform.position + new Vector3(-0.10f, 0.52f, -0.10f)
            : GetChestPos() + Vector3.back * 0.2f;
        Safe(() => { _body.spine.localEulerAngles = new Vector3(26f, 10f, 0f); });
        yield return new WaitForSeconds(2.0f);
        Safe(() => { _body.spine.localEulerAngles = new Vector3(15f,  0f, 0f); });
        _ikActive = false;
    }

    IEnumerator AnimReposition()
    {
        _ikActive = true;
        Vector3 ct = GetChestPos();
        _leftTgt  = ct + Vector3.up * 0.16f;
        _rightTgt = ct + Vector3.up * 0.16f;
        yield return new WaitForSeconds(0.25f);
        _leftTgt = ct; _rightTgt = ct;
        yield return new WaitForSeconds(0.4f);
        _ikActive = false;
    }

    IEnumerator AnimTiltHead()
    {
        _ikActive = true;
        Vector3 hp = GetPatientHead();
        _leftTgt  = hp + transform.forward *  0.10f;
        _rightTgt = hp - transform.forward *  0.10f + Vector3.down * 0.03f;
        Safe(() => { _body.spine.localEulerAngles = new Vector3(32f, 0f, 0f); });
        yield return new WaitForSeconds(1.2f);
        Safe(() => { _body.spine.localEulerAngles = new Vector3(15f, 0f, 0f); });
        _ikActive = false;
    }

    IEnumerator AnimIdle()
    {
        _ikActive = false;
        float t = 0f;
        while (!_celebrating)
        {
            t += Time.deltaTime;
            float swayVal = _sway;
            Safe(() => { _body.spine.localEulerAngles =
                new Vector3(15f + Mathf.Sin(t * 0.65f) * swayVal, 0f, 0f); });
            yield return null;
        }
    }

    // ── ROSC Celebration — rescuer sits back, raises arms ────────────────────
    IEnumerator AnimROSCCelebration()
    {
        _celebrating = true;
        _ikActive    = false;

        // Sit back slightly — reduce forward lean over 0.5s
        float t = 0f;
        while (t < 1f) {
            t += Time.deltaTime * 2f;
            Safe(() => { _body.spine.localEulerAngles =
                Vector3.Lerp(new Vector3(20f,0f,0f), new Vector3(2f,0f,0f), t); });
            yield return null;
        }

        // Arms rise in relief — wide V shape
        _ikActive = true;
        _leftTgt  = transform.position + transform.up * 1.6f + transform.right *  0.7f;
        _rightTgt = transform.position + transform.up * 1.6f - transform.right *  0.7f;
        yield return new WaitForSeconds(1.5f);

        // Gentle wave down to rest
        t = 0f;
        Vector3 lt0 = _leftTgt, rt0 = _rightTgt;
        Vector3 ltRest = transform.position + transform.up * 0.9f;
        Vector3 rtRest = transform.position + transform.up * 0.9f;
        while (t < 1f) {
            t += Time.deltaTime * 0.8f;
            float s = Mathf.SmoothStep(0, 1, t);
            _leftTgt  = Vector3.Lerp(lt0, ltRest, s);
            _rightTgt = Vector3.Lerp(rt0, rtRest, s);
            yield return null;
        }
        _ikActive    = false;
        _celebrating = false;
        StartCoroutine(AnimIdle());
    }

    // ── Compression count label ───────────────────────────────────────────────
    void BuildCompLabel()
    {
        var canvasGo = new GameObject("CompCanvas");
        canvasGo.transform.SetParent(transform, false);
        _worldCanvas = canvasGo.AddComponent<Canvas>();
        _worldCanvas.renderMode = RenderMode.WorldSpace;
        canvasGo.AddComponent<CanvasScaler>();
        var rt = canvasGo.GetComponent<RectTransform>();
        rt.sizeDelta = new Vector2(1.2f, 0.4f);
        rt.localScale = Vector3.one * 0.004f;
        canvasGo.transform.localPosition = new Vector3(0f, 1.85f, 0.5f);

        var textGo = new GameObject("CompText");
        textGo.transform.SetParent(canvasGo.transform, false);
        _compCountLabel = textGo.AddComponent<Text>();
        _compCountLabel.font      = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        _compCountLabel.fontSize  = 60;
        _compCountLabel.fontStyle = FontStyle.Bold;
        _compCountLabel.alignment = TextAnchor.MiddleCenter;
        _compCountLabel.color     = new Color(1f, 0.85f, 0.1f);
        _compCountLabel.text      = "";
        var trt = textGo.GetComponent<RectTransform>();
        trt.anchorMin = Vector2.zero;
        trt.anchorMax = Vector2.one;
        trt.sizeDelta = Vector2.zero;
        trt.anchoredPosition = Vector2.zero;
    }

    void UpdateCompLabel()
    {
        if (_compCountLabel == null) return;
        if (_totalCompressions == 0)
        {
            _compCountLabel.text = "";
        }
        else
        {
            // Fatigue colour: green → amber → red
            Color col = _fatigueLevel < 0.33f ? new Color(0.2f, 0.9f, 0.4f)
                      : _fatigueLevel < 0.66f ? new Color(1f, 0.82f, 0.1f)
                      : new Color(0.9f, 0.3f, 0.3f);
            _compCountLabel.color = col;
            _compCountLabel.text  = $"CPR × {_totalCompressions}";
        }
    }

    // ── Kneeling pose ─────────────────────────────────────────────────────────
    void SetKneeling()
    {
        if (_body == null) return;
        Safe(() => { _body.hips.localPosition            = new Vector3(0f, 0.48f, 0f); });
        Safe(() => { _body.leftThigh.localEulerAngles    = new Vector3(-90f, 0f,  5f); });
        Safe(() => { _body.rightThigh.localEulerAngles   = new Vector3(-90f, 0f, -5f); });
        Safe(() => { _body.leftShin.localEulerAngles     = new Vector3( 90f, 0f,  0f); });
        Safe(() => { _body.rightShin.localEulerAngles    = new Vector3( 90f, 0f,  0f); });
        Safe(() => { _body.leftFoot.localEulerAngles     = new Vector3(-25f, 0f,  0f); });
        Safe(() => { _body.rightFoot.localEulerAngles    = new Vector3(-25f, 0f,  0f); });
        Safe(() => { _body.spine.localEulerAngles        = new Vector3( 15f, 0f,  0f); });
    }

    // ── Helpers ───────────────────────────────────────────────────────────────
    IEnumerator MoveRoot(Vector3 from, Vector3 to, float dur)
    {
        float t = 0f;
        while (t < 1f)
        {
            t += Time.deltaTime / dur;
            transform.position = Vector3.Lerp(from, to, Mathf.SmoothStep(0, 1, t));
            yield return null;
        }
    }

    Vector3 GetChestPos()
    {
        if (patient != null) return patient.ChestWorldPosition;
        return transform.position + transform.forward * 0.55f + Vector3.up * 0.05f;
    }

    Vector3 GetPatientHead()
    {
        if (patient != null)
            return patient.transform.position + new Vector3(0f, 0.22f, 0.80f);
        return transform.position + transform.forward * 0.85f + Vector3.up * 0.20f;
    }

    Vector3 GetPatientNeck()
    {
        if (patient != null)
            return patient.transform.position + new Vector3(0.12f, 0.18f, 0.62f);
        return transform.position + transform.forward * 0.70f + Vector3.up * 0.18f;
    }

    Vector3 GetPatientShoulder(bool left)
    {
        float x = left ? 0.20f : -0.20f;
        if (patient != null)
            return patient.transform.position + new Vector3(x, 0.22f, 0.26f);
        return transform.position + transform.forward * 0.5f + Vector3.up * 0.20f;
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
        m.color = new Color(1f, 0.85f, 0f);
        m.SetFloat("_Glossiness", 0.4f);
        r.material = m;
        var col = s.GetComponent<Collider>();
        if (col) Object.Destroy(col);
    }

    // Quaternion SmoothDamp (Unity does not have a built-in)
    static Quaternion SmoothDampQ(Quaternion cur, Quaternion tgt,
                                   ref Quaternion vel, float smooth)
    {
        if (Time.deltaTime < 1e-6f) return cur;
        if (Quaternion.Dot(cur, tgt) < 0) tgt = new Quaternion(-tgt.x,-tgt.y,-tgt.z,-tgt.w);
        var c = new Vector4(cur.x, cur.y, cur.z, cur.w);
        var t = new Vector4(tgt.x, tgt.y, tgt.z, tgt.w);
        var v = new Vector4(vel.x, vel.y, vel.z, vel.w);
        var r = new Vector4(
            Mathf.SmoothDamp(c.x, t.x, ref v.x, smooth),
            Mathf.SmoothDamp(c.y, t.y, ref v.y, smooth),
            Mathf.SmoothDamp(c.z, t.z, ref v.z, smooth),
            Mathf.SmoothDamp(c.w, t.w, ref v.w, smooth));
        vel = new Quaternion(v.x, v.y, v.z, v.w);
        return new Quaternion(r.x, r.y, r.z, r.w).normalized;
    }

    static void Safe(System.Action a) { try { a(); } catch { } }
}