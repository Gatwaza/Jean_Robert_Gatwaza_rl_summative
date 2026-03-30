// CPR_HUD.cs
// ===========
// Layperson-facing HUD displayed as a Unity Canvas overlay.
// Shows: phase banner, feedback text, vital signs, action guide,
//        reward progress, training experiment tracker.
//
// Requires: Unity UI (Canvas + Text/Image components).
// This script uses Unity's legacy UI system (compatible with all Unity versions).
// For TextMeshPro, swap Text references to TMP_Text.
//
// Canvas Setup:
//   Create a Canvas (Screen Space - Overlay).
//   Attach all UI element references in Inspector.

using System.Collections;
using UnityEngine;
using UnityEngine.UI;

public class CPR_HUD : MonoBehaviour
{
    // ── Header panel ─────────────────────────────────────────────────────────
    [Header("Header")]
    public Text    phaseLabel;         // "TRAINING — DQN  Experiment 3/10"
    public Text    connectionStatus;   // "● Connected" / "○ Waiting for Python..."
    public Image   phaseBanner;

    // ── Feedback panel (large, centre-bottom) ─────────────────────────────
    [Header("Feedback")]
    public Text    feedbackText;       // "✓ Good! Push hard and fast."
    public Image   feedbackPanel;
    public Image   correctIndicator;   // green tick
    public Image   incorrectIndicator; // red cross

    // ── Action display ────────────────────────────────────────────────────
    [Header("Action")]
    public Text    actionNameText;     // "BEGIN CHEST COMPRESSIONS"
    public Text    stepCounter;        // "Step 42 / Episode 3"
    public Text    rewardText;         // "Reward: +2.85   Total: 14.35"

    // ── Vital signs panel ────────────────────────────────────────────────
    [Header("Vitals")]
    public Slider  heartRateSlider;
    public Slider  chestRiseSlider;
    public Slider  handPlacementSlider;
    public Slider  consciousnessSlider;
    public Text    heartRateLabel;
    public Text    airwayStatusText;
    public Text    compressionsText;
    public Text    roscText;

    // ── Training progress ────────────────────────────────────────────────
    [Header("Training Progress")]
    public Text    experimentText;     // "Experiment 7 / 10"
    public Slider  trainingProgress;
    public Text    meanRewardText;
    public Text    algorithmText;

    // ── Protocol guide (step-by-step instruction panel) ──────────────────
    [Header("Protocol Guide")]
    public GameObject protocolGuidePanel;
    public Text[]     protocolSteps;   // 6 Text elements for CPR stages
    public Image[]    protocolTicks;   // green tick per completed stage

    // ── ROSC celebration ────────────────────────────────────────────────
    [Header("ROSC")]
    public GameObject roscPanel;       // "PATIENT REVIVED ★" fullscreen overlay

    // ── Colours ──────────────────────────────────────────────────────────
    private static readonly Color ColCorrect   = new Color(0.13f, 0.75f, 0.36f);
    private static readonly Color ColIncorrect = new Color(0.85f, 0.22f, 0.22f);
    private static readonly Color ColNeutral   = new Color(0.15f, 0.18f, 0.25f, 0.85f);
    private static readonly Color ColRandom    = new Color(0.4f, 0.4f, 0.55f);
    private static readonly Color ColTraining  = new Color(0.18f, 0.35f, 0.62f);
    private static readonly Color ColDemo      = new Color(0.12f, 0.48f, 0.22f);

    // Protocol stage labels for the guide panel
    private static readonly string[] StagLabels =
    {
        "1. Assess & Call 911",
        "2. Open Airway",
        "3. Check Breathing",
        "4. Position Hands",
        "5. Chest Compressions",
        "6. Rescue Breaths / AED",
    };

    private float _feedbackTimer = 0f;
    private int   _completedStages = 0;

    // ── Unity lifecycle ───────────────────────────────────────────────────────

    void Awake()
    {
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

    void Start()
    {
        InitProtocolGuide();
        SetConnectionStatus(false);
        ShowROSC(false);
        if (feedbackPanel != null)
            feedbackPanel.color = ColNeutral;
    }

    void Update()
    {
        // Fade feedback panel back to neutral after display duration
        if (_feedbackTimer > 0f)
        {
            _feedbackTimer -= Time.deltaTime;
            if (_feedbackTimer <= 0f && feedbackPanel != null)
                StartCoroutine(FadePanelColor(feedbackPanel, ColNeutral, 0.5f));
        }
    }

    // ── Event handlers ────────────────────────────────────────────────────────

    private void HandleState(StatePacket p)
    {
        // Action display
        SetText(actionNameText, p.action_name?.Replace("_", " ") ?? "");
        SetText(stepCounter,    $"Step {p.step}   |   Episode {p.episode}");
        SetText(rewardText,     $"Reward: {p.reward:+0.00;-0.00}   Total: {p.cumulative_reward:0.0}");

        // Vitals
        SetSlider(heartRateSlider,      p.vitals?.heart_rate ?? 0f);
        SetSlider(chestRiseSlider,      p.vitals?.chest_rise ?? 0f);
        SetSlider(handPlacementSlider,  p.vitals?.hand_placement ?? 0f);
        SetSlider(consciousnessSlider,  p.vitals?.consciousness ?? 0f);
        SetText(heartRateLabel,   $"HR: {(p.vitals?.heart_rate ?? 0f)*100:0}%");
        SetText(airwayStatusText, p.vitals?.airway_open == true ? "Airway: OPEN ✓" : "Airway: CLOSED ✗");
        SetText(compressionsText, $"Compressions: {p.vitals?.compressions ?? 0}");

        // ROSC
        if (p.rosc)
        {
            SetText(roscText, "★ ROSC — Patient Responding!");
            ShowROSC(true);
        }

        // Training progress
        SetText(meanRewardText,   $"Mean Reward: {p.mean_reward:0.0}");
        SetText(experimentText,   $"Experiment {p.experiment} / 10");
        SetSlider(trainingProgress, p.experiment / 10f);

        // Feedback (big centre-bottom instruction for layperson)
        ShowFeedback(p.feedback, p.is_correct);

        // Protocol guide — tick completed stages
        UpdateProtocolTicks(p.protocol_stage);

        // Correct/incorrect indicators
        if (correctIndicator != null)   correctIndicator.gameObject.SetActive(p.is_correct);
        if (incorrectIndicator != null) incorrectIndicator.gameObject.SetActive(!p.is_correct);
    }

    private void HandlePhaseChange(StatePacket p)
    {
        string phaseName = p.phase switch
        {
            "random"   => "EXPLORING",
            "training" => $"TRAINING  {p.algorithm}",
            "demo"     => $"DEMO  {p.algorithm}  Best Model",
            _          => p.phase.ToUpper(),
        };
        SetText(phaseLabel,    phaseName);
        SetText(algorithmText, p.algorithm ?? "");

        Color bannerColor = p.phase switch
        {
            "random"   => ColRandom,
            "training" => ColTraining,
            "demo"     => ColDemo,
            _          => ColNeutral,
        };
        if (phaseBanner != null)
            StartCoroutine(FadePanelColor(phaseBanner, bannerColor, 0.8f));

        _completedStages = 0;
        UpdateProtocolTicks(0);
        ShowROSC(false);
    }

    private void HandleEpisodeEnd(StatePacket p)
    {
        if (p.rosc)
            ShowROSC(true);
        else
            ShowROSC(false);
    }

    private void HandleConnection(bool connected)
    {
        SetConnectionStatus(connected);
    }

    // ── UI helpers ────────────────────────────────────────────────────────────

    private void ShowFeedback(string text, bool correct)
    {
        SetText(feedbackText, text);
        Color col = correct ? ColCorrect : ColIncorrect;
        if (feedbackPanel != null)
        {
            feedbackPanel.color = col;
        }
        _feedbackTimer = 2.5f;  // show for 2.5s then fade
    }

    private void UpdateProtocolTicks(int stage)
    {
        _completedStages = Mathf.Max(_completedStages, stage);
        if (protocolTicks == null) return;
        for (int i = 0; i < protocolTicks.Length; i++)
        {
            if (protocolTicks[i] != null)
                protocolTicks[i].gameObject.SetActive(i < _completedStages);
        }
    }

    private void InitProtocolGuide()
    {
        if (protocolSteps == null) return;
        for (int i = 0; i < protocolSteps.Length && i < StagLabels.Length; i++)
            SetText(protocolSteps[i], StagLabels[i]);
        UpdateProtocolTicks(0);
    }

    private void ShowROSC(bool show)
    {
        if (roscPanel != null) roscPanel.SetActive(show);
    }

    private void SetConnectionStatus(bool connected)
    {
        SetText(connectionStatus, connected
            ? "<color=#2ecc71>● Connected to RL Engine</color>"
            : "<color=#e74c3c>○ Waiting for Python...</color>");
    }

    private static void SetText(Text t, string s)
    {
        if (t != null) t.text = s;
    }

    private static void SetSlider(Slider s, float val)
    {
        if (s != null) s.value = Mathf.Clamp01(val);
    }

    private IEnumerator FadePanelColor(Image img, Color target, float duration)
    {
        if (img == null) yield break;
        Color start = img.color;
        float t = 0f;
        while (t < 1f)
        {
            t += Time.deltaTime / duration;
            img.color = Color.Lerp(start, target, t);
            yield return null;
        }
    }
}
