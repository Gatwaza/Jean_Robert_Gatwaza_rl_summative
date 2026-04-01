// CPRSceneManager.cs (v5 — stable camera facing rescuer, ROSC/fail indicator)
// Camera stays FIXED during action animations; only moves on phase changes.
// Per-episode outcome panel shown on Game view.

using System.Collections;
using UnityEngine;
using UnityEngine.UI;

public class CPRSceneManager : MonoBehaviour
{
    [Header("Auto-created at runtime")]
    public HumanoidPatient     patient;
    public HumanoidRescuer     rescuer;
    public CPRSceneEnvironment environment;
    public CPR_HUD             hud;

    [Header("Camera")]
    public Camera mainCamera;

    [Header("Outcome Panel (assign or auto-created)")]
    public GameObject outcomePanelGO;
    public Text       outcomeText;

    // ── Single stable camera position — always sees both avatars ─────────────
    // Positioned at slight left-front of rescuer, angled across the scene.
    // This never changes during an episode — only slides gently on phase change.
    private static readonly Vector3 StableCamPos = new Vector3(-2.1f, 1.45f, -0.9f);
    private static readonly Vector3 StableLookAt = new Vector3( 0.28f, 0.30f,  0.2f);
    private static readonly float   StableFOV    = 52f;

    // Spring camera (only used for phase change drift)
    private Vector3    _camVelPos = Vector3.zero;
    private Vector3    _camTgtPos = StableCamPos;
    private Quaternion _camTgtRot;
    private float      _camVelFov = 0f;
    private float      _tgtFov    = StableFOV;
    private float      _smooth    = 0.5f;

    void Start()
    {
        if (mainCamera == null) mainCamera = Camera.main;
        SpawnScene();
        BuildOutcomePanel();

        // Set camera immediately to stable position
        if (mainCamera != null)
        {
            mainCamera.transform.position = StableCamPos;
            mainCamera.transform.rotation = Quaternion.LookRotation(
                (StableLookAt - StableCamPos).normalized, Vector3.up);
            mainCamera.fieldOfView = StableFOV;
        }
        _camTgtPos = StableCamPos;
        _camTgtRot = Quaternion.LookRotation((StableLookAt - StableCamPos).normalized, Vector3.up);

        TransitionPhase("random", "RANDOM");

        BridgeEvents.OnStateUpdate       += HandleState;
        BridgeEvents.OnPhaseChange       += HandlePhaseChange;
        BridgeEvents.OnEpisodeEnd        += HandleEpisodeEnd;
        BridgeEvents.OnConnectionChanged += HandleConnection;
    }

    void OnDestroy()
    {
        BridgeEvents.OnStateUpdate       -= HandleState;
        BridgeEvents.OnPhaseChange       -= HandlePhaseChange;
        BridgeEvents.OnEpisodeEnd        -= HandleEpisodeEnd;
        BridgeEvents.OnConnectionChanged -= HandleConnection;
    }

    void Update()
    {
        if (mainCamera == null) return;
        // Very gentle SmoothDamp — only noticeable on phase changes
        mainCamera.transform.position = Vector3.SmoothDamp(
            mainCamera.transform.position, _camTgtPos, ref _camVelPos, _smooth);
        mainCamera.transform.rotation = Quaternion.Slerp(
            mainCamera.transform.rotation, _camTgtRot, Time.deltaTime/_smooth*0.5f);
        mainCamera.fieldOfView = Mathf.SmoothDamp(
            mainCamera.fieldOfView, _tgtFov, ref _camVelFov, _smooth*0.8f);
    }

    void SpawnScene()
    {
        var envGo = new GameObject("Environment");
        environment = envGo.AddComponent<CPRSceneEnvironment>();

        var patGo = new GameObject("Patient");
        patGo.transform.position = Vector3.zero;
        patient = patGo.AddComponent<HumanoidPatient>();

        var resGo = new GameObject("Rescuer");
        resGo.transform.position = new Vector3(0.68f, 0f, 0.12f);
        resGo.transform.rotation = Quaternion.Euler(0f, -90f, 0f);
        rescuer = resGo.AddComponent<HumanoidRescuer>();
        rescuer.patient = patient;

        var canvasGo = GameObject.Find("Canvas");
        if (canvasGo != null) hud = canvasGo.GetComponent<CPR_HUD>();

        // ── Auto-build live action ticker overlay ─────────────────────────────
        // Works even when Canvas UI fields are not wired in Inspector.
        BuildLiveActionTicker(canvasGo);
    }

    // ── Live action ticker — shows current action like the terminal ───────────
    private UnityEngine.UI.Text _actionTicker;
    private UnityEngine.UI.Text _stepTicker;
    private UnityEngine.UI.Text _hrTicker;

    void BuildLiveActionTicker(GameObject canvas)
    {
        if (canvas == null) return;

        // Semi-transparent black strip across top
        var strip = new GameObject("ActionStrip");
        strip.transform.SetParent(canvas.transform, false);
        var img = strip.AddComponent<UnityEngine.UI.Image>();
        img.color = new Color(0f, 0f, 0f, 0.72f);
        var rt = strip.GetComponent<RectTransform>();
        rt.anchorMin = new Vector2(0f, 1f); rt.anchorMax = new Vector2(1f, 1f);
        rt.pivot     = new Vector2(0.5f, 1f);
        rt.sizeDelta = new Vector2(0f, 50f);
        rt.anchoredPosition = Vector2.zero;

        // Action name — large, centred
        _actionTicker = MakeText(strip, "ActionTick",
            new Vector2(0f, 0f), new Vector2(0.55f, 1f),
            20, FontStyle.Bold, TextAnchor.MiddleLeft, new Vector2(12f, 0f));

        // Step / episode info — right side
        _stepTicker = MakeText(strip, "StepTick",
            new Vector2(0.55f, 0f), new Vector2(0.78f, 1f),
            14, FontStyle.Normal, TextAnchor.MiddleCenter, Vector2.zero);

        // HR — rightmost
        _hrTicker = MakeText(strip, "HRTick",
            new Vector2(0.78f, 0f), new Vector2(1.0f, 1f),
            14, FontStyle.Bold, TextAnchor.MiddleCenter, Vector2.zero);
    }

    UnityEngine.UI.Text MakeText(GameObject parent, string name,
                                  Vector2 ancMin, Vector2 ancMax,
                                  int size, FontStyle style, TextAnchor anchor,
                                  Vector2 offset)
    {
        var go = new GameObject(name);
        go.transform.SetParent(parent.transform, false);
        var t = go.AddComponent<UnityEngine.UI.Text>();
        t.font      = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        t.fontSize  = size;
        t.fontStyle = style;
        t.alignment = anchor;
        t.color     = Color.white;
        t.text      = "";
        var rt = go.GetComponent<RectTransform>();
        rt.anchorMin = ancMin; rt.anchorMax = ancMax;
        rt.offsetMin = offset; rt.offsetMax = Vector2.zero;
        rt.sizeDelta = Vector2.zero;
        return t;
    }

    void UpdateLiveTicker(StatePacket p)
    {
        if (_actionTicker != null)
            _actionTicker.text = $"  ▶  {(p.action_name ?? "").Replace("_"," ")}";

        if (_stepTicker != null)
            _stepTicker.text = $"Step {p.step}  |  Ep {p.episode}  |  "
                             + $"R: {p.reward:+0.00;-0.00}  Total: {p.cumulative_reward:0.0}";

        float hr = p.vitals?.heart_rate ?? 0f;
        if (_hrTicker != null)
        {
            _hrTicker.color = hr >= 0.9f ? new Color(0.2f,0.9f,0.3f)
                            : hr >= 0.5f ? new Color(1f,0.85f,0.1f)
                            : new Color(0.9f,0.3f,0.3f);
            _hrTicker.text = $"HR: {hr*100:0}%{(p.rosc ? "  ★ ROSC" : "")}";
        }
    }

    // ── Outcome panel — shown at episode end ──────────────────────────────────
    void BuildOutcomePanel()
    {
        if (outcomePanelGO != null) return;   // already assigned in Inspector

        var canvas = GameObject.Find("Canvas");
        if (canvas == null) return;

        // Panel
        outcomePanelGO = new GameObject("OutcomePanel");
        outcomePanelGO.transform.SetParent(canvas.transform, false);

        var img = outcomePanelGO.AddComponent<Image>();
        img.color = new Color(0,0,0,0);   // start invisible

        var rect = outcomePanelGO.GetComponent<RectTransform>();
        rect.anchorMin = new Vector2(0.5f, 0.05f);
        rect.anchorMax = new Vector2(0.5f, 0.05f);
        rect.pivot     = new Vector2(0.5f, 0f);
        rect.sizeDelta = new Vector2(320f, 60f);
        rect.anchoredPosition = Vector2.zero;

        // Text
        var textGo = new GameObject("OutcomeText");
        textGo.transform.SetParent(outcomePanelGO.transform, false);
        outcomeText = textGo.AddComponent<Text>();
        outcomeText.font      = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        outcomeText.fontSize  = 22;
        outcomeText.fontStyle = FontStyle.Bold;
        outcomeText.alignment = TextAnchor.MiddleCenter;
        outcomeText.text      = "";
        var tr = textGo.GetComponent<RectTransform>();
        tr.anchorMin  = Vector2.zero; tr.anchorMax = Vector2.one;
        tr.sizeDelta  = Vector2.zero; tr.anchoredPosition = Vector2.zero;

        outcomePanelGO.SetActive(false);
    }

    void ShowOutcome(bool rosc, int episode)
    {
        if (outcomePanelGO == null || outcomeText == null) return;
        outcomePanelGO.SetActive(true);
        var img = outcomePanelGO.GetComponent<Image>();
        if (rosc)
        {
            img.color = new Color(0.10f, 0.55f, 0.15f, 0.88f);
            outcomeText.color = Color.white;
            outcomeText.text  = $"Episode {episode}  —  PATIENT REVIVED";
        }
        else
        {
            img.color = new Color(0.60f, 0.10f, 0.10f, 0.88f);
            outcomeText.color = Color.white;
            outcomeText.text  = $"Episode {episode}  —  NOT REVIVED";
        }
        StartCoroutine(FadeOutPanel(3.5f));
    }

    IEnumerator FadeOutPanel(float delay)
    {
        yield return new WaitForSeconds(delay);
        if (outcomePanelGO == null) yield break;
        var img = outcomePanelGO.GetComponent<Image>();
        float t = 0f;
        Color startImg  = img.color;
        Color startTxt  = outcomeText.color;
        while (t < 1f)
        {
            t += Time.deltaTime * 1.2f;
            img.color         = Color.Lerp(startImg, new Color(startImg.r,startImg.g,startImg.b,0), t);
            outcomeText.color = Color.Lerp(startTxt, new Color(1,1,1,0), t);
            yield return null;
        }
        outcomePanelGO.SetActive(false);
    }

    // ── Event handlers ────────────────────────────────────────────────────────
    void HandleState(StatePacket p)
    {
        UpdateLiveTicker(p);
        // CAMERA DOES NOT MOVE on actions — stays at stable position
        if (p.rosc) environment?.FlashROSC();
        else if (!p.is_correct) environment?.FlashIncorrect();
    }

    void HandlePhaseChange(StatePacket p) => TransitionPhase(p.phase, p.algorithm);

    void HandleEpisodeEnd(StatePacket p)
    {
        ShowOutcome(p.rosc, p.episode);
        if (p.rosc) environment?.FlashROSC();
    }

    void HandleConnection(bool connected)
    {
        if (environment?.ambientFill != null)
            environment.ambientFill.intensity = connected ? 0.55f : 0.18f;
    }

    // ── Phase transitions — only time camera moves ────────────────────────────
    void TransitionPhase(string phase, string algo)
    {
        Color col = phase switch {
            "training" => new Color(0.38f,0.48f,0.78f),
            "demo"     => new Color(0.30f,0.68f,0.38f),
            _          => new Color(0.52f,0.52f,0.58f)
        };
        environment?.SetPhaseAmbient(col);

        // Slight camera position variation per phase — but always frontal
        _camTgtPos = StableCamPos + phase switch {
            "random"   => new Vector3(0.3f,  0.2f, 0.5f),   // wider overhead-ish
            "training" => new Vector3(0.0f,  0.0f, 0.0f),   // standard
            "demo"     => new Vector3(-0.2f,-0.1f,-0.3f),   // slightly tighter
            _          => Vector3.zero
        };
        _camTgtRot = Quaternion.LookRotation(
            (StableLookAt - _camTgtPos).normalized, Vector3.up);
        _tgtFov = phase == "random" ? 58f : phase == "demo" ? 48f : StableFOV;
        _smooth = 1.8f;
        StartCoroutine(RestoreSmooth(2.5f, 0.5f));
    }

    IEnumerator RestoreSmooth(float delay, float target)
    {
        yield return new WaitForSeconds(delay);
        _smooth = target;
    }
}