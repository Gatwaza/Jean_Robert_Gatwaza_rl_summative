// CPRSceneManager.cs  (v4 — tuned cameras, smooth spring, realistic pacing)
// ===========================================================================
// Camera rigs redesigned around the actual layout:
//   Patient lies along +Z axis, head near z=0.8, feet near z=-0.8
//   Rescuer kneels at world (0.65, 0, 0.1), faces -X (toward patient)
//   Scene centre of interest: (0.25, 0.3, 0.15)
//
// All positions verified to keep both avatars visible simultaneously.

using System.Collections;
using UnityEngine;

public class CPRSceneManager : MonoBehaviour
{
    [Header("Auto-created at runtime")]
    public HumanoidPatient     patient;
    public HumanoidRescuer     rescuer;
    public CPRSceneEnvironment environment;
    public CPR_HUD             hud;

    [Header("Camera")]
    public Camera mainCamera;

    // ── Camera rig definitions ────────────────────────────────────────────────
    // All positions hand-tuned so both avatars are in frame with good composition.
    // Look-at target is the midpoint between rescuer torso and patient chest.
    struct CamRig { public Vector3 pos; public Vector3 lookAt; public float fov; }

    // ── CAMERA RIGS — coordinate reference ────────────────────────────────────
    // Patient: head at z≈+0.85, feet at z≈-0.85, body at y≈0.14, x≈0
    // Rescuer: kneels at x=0.68, z=0.12, torso at y≈0.6, faces -X direction
    // Scene interest point: (0.25, 0.35, 0.1) — midpoint rescuer/patient chest
    //
    // Rule: camera must see BOTH avatars. Put it on the OPPOSITE side from
    // the rescuer (rescuer is at +X, so camera goes to -X side) and look
    // across the scene. The patient lies along Z so a side view (-X, looking +X)
    // shows the full body length with rescuer kneeling in frame.

    // Default 3/4 view: camera at rescuer's back-left, looking diagonally
    // across scene. Both avatars visible, rescuer's arms and patient both clear.
    static readonly CamRig RigStoryteller = new CamRig {
        pos    = new Vector3(-2.2f, 1.4f, -0.8f),  // behind patient's shoulders, offset left
        lookAt = new Vector3( 0.3f, 0.3f,  0.2f),  // patient chest / rescuer hands
        fov    = 54f
    };

    // Side-on compression view: pure -X side, low to show weight transfer.
    // Patient body horizontal, rescuer arms driving down — best silhouette.
    static readonly CamRig RigCompression = new CamRig {
        pos    = new Vector3(-2.8f, 0.6f,  0.1f),
        lookAt = new Vector3( 0.2f, 0.3f,  0.1f),
        fov    = 42f
    };

    // Airway close-up: camera at patient's feet (+Z), looking toward head.
    // Shows rescuer cupping head, neck extension clearly.
    static readonly CamRig RigAirway = new CamRig {
        pos    = new Vector3( 0.2f, 0.8f, -1.6f),  // at patient's feet
        lookAt = new Vector3( 0.1f, 0.18f, 0.7f),  // toward patient's head
        fov    = 40f
    };

    // Overhead: straight down, shows full mat, both avatars.
    static readonly CamRig RigOverhead = new CamRig {
        pos    = new Vector3( 0.3f, 4.2f,  0.1f),
        lookAt = new Vector3( 0.3f, 0.0f,  0.1f),
        fov    = 62f
    };

    // ROSC cinematic: low angle from -X side near patient's head,
    // looking up — rescuer silhouetted against ceiling light.
    static readonly CamRig RigROSC = new CamRig {
        pos    = new Vector3(-1.4f, 0.25f, 0.8f),
        lookAt = new Vector3( 0.3f, 0.7f,  0.1f),
        fov    = 44f
    };

    // Training: wider pull-back, same angle as storyteller.
    static readonly CamRig RigTraining = new CamRig {
        pos    = new Vector3(-3.0f, 1.8f, -1.2f),
        lookAt = new Vector3( 0.3f, 0.25f, 0.1f),
        fov    = 58f
    };

    // ── Action → rig ─────────────────────────────────────────────────────────
    static readonly System.Collections.Generic.Dictionary<int, string> ActionRig
        = new() {
        {0,"storyteller"}, {1,"overhead"},    {2,"airway"},      {3,"airway"},
        {4,"compression"}, {5,"airway"},      {6,"overhead"},    {7,"airway"},
        {8,"storyteller"}, {9,"compression"}, {10,"airway"},     {11,"storyteller"},
    };

    // ── Camera state ─────────────────────────────────────────────────────────
    private Vector3    _camVelPos  = Vector3.zero;
    private Vector3    _camTgtPos;
    private Quaternion _camTgtRot;
    private float      _camVelFov  = 0f;
    private float      _tgtFov     = 50f;
    private float      _smooth     = 0.6f;

    // Orbital camera
    private float  _orbitAngle      = -35f;   // start angle (degrees around Y)
    private float  _orbitRadius     = 2.8f;   // distance from scene centre
    private float  _orbitHeight     = 1.35f;  // elevation
    private float  _orbitSpeed      = 12f;    // degrees per second
    private bool   _orbiting        = true;
    private float  _orbitResumeT    = 0f;     // time to resume orbit after cut
    private static readonly Vector3 SceneCentre = new Vector3(0.3f, 0.3f, 0.1f);

    void Start()
    {
        if (mainCamera == null) mainCamera = Camera.main;
        SpawnScene();
        ApplyRig(RigTraining, instant: true);
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

        // Resume orbit after a cut has settled
        if (!_orbiting && Time.time > _orbitResumeT)
            _orbiting = true;

        if (_orbiting)
        {
            // Advance orbital angle
            _orbitAngle += _orbitSpeed * Time.deltaTime;
            if (_orbitAngle > 360f) _orbitAngle -= 360f;

            // Compute position on circle around SceneCentre
            float rad = _orbitAngle * Mathf.Deg2Rad;
            Vector3 orbitPos = SceneCentre + new Vector3(
                Mathf.Sin(rad) * _orbitRadius,
                _orbitHeight,
                Mathf.Cos(rad) * _orbitRadius);

            // Smooth drift toward orbit position — not snappy
            mainCamera.transform.position = Vector3.SmoothDamp(
                mainCamera.transform.position, orbitPos, ref _camVelPos, 0.8f);
            Quaternion lookRot = Quaternion.LookRotation(
                (SceneCentre - mainCamera.transform.position).normalized, Vector3.up);
            mainCamera.transform.rotation = Quaternion.Slerp(
                mainCamera.transform.rotation, lookRot, Time.deltaTime * 1.8f);
        }
        else
        {
            // Driven cut — spring to _camTgtPos
            mainCamera.transform.position = Vector3.SmoothDamp(
                mainCamera.transform.position, _camTgtPos, ref _camVelPos, _smooth);
            mainCamera.transform.rotation = Quaternion.Slerp(
                mainCamera.transform.rotation, _camTgtRot,
                Time.deltaTime / Mathf.Max(_smooth * 0.7f, 0.01f));
        }

        mainCamera.fieldOfView = Mathf.SmoothDamp(
            mainCamera.fieldOfView, _tgtFov, ref _camVelFov, _smooth * 0.8f);
    }

    void SpawnScene()
    {
        var envGo = new GameObject("Environment");
        environment = envGo.AddComponent<CPRSceneEnvironment>();

        // Patient — root at origin, script lifts to y=0.14 internally
        var patGo = new GameObject("Patient");
        patGo.transform.position = Vector3.zero;
        patient = patGo.AddComponent<HumanoidPatient>();

        // Rescuer — kneels to the right (+X) of patient, faces toward -X
        var resGo = new GameObject("Rescuer");
        resGo.transform.position = new Vector3(0.68f, 0f, 0.12f);
        resGo.transform.rotation = Quaternion.Euler(0f, -90f, 0f);
        rescuer = resGo.AddComponent<HumanoidRescuer>();
        rescuer.patient = patient;

        var canvasGo = GameObject.Find("Canvas");
        if (canvasGo != null) hud = canvasGo.GetComponent<CPR_HUD>();
    }

    // ── Events ────────────────────────────────────────────────────────────────

    void HandleState(StatePacket p)
    {
        if (ActionRig.TryGetValue(p.action, out string rigName))
        {
            _smooth = 0.5f;    // responsive but not snappy between actions
            SetCameraByName(rigName);
        }
        if (p.rosc)
        {
            environment?.FlashROSC();
            StartCoroutine(ROSCCameraSequence());
        }
        else if (!p.is_correct)
        {
            environment?.FlashIncorrect();
        }
    }

    void HandlePhaseChange(StatePacket p) => TransitionPhase(p.phase, p.algorithm);
    void HandleEpisodeEnd(StatePacket p)
    {
        if (p.rosc) StartCoroutine(ROSCCameraSequence());
    }
    void HandleConnection(bool connected)
    {
        if (environment?.ambientFill != null)
            environment.ambientFill.intensity = connected ? 0.55f : 0.18f;
    }

    void TransitionPhase(string phase, string algo)
    {
        Color col = phase switch {
            "training" => new Color(0.38f, 0.48f, 0.78f),
            "demo"     => new Color(0.30f, 0.68f, 0.38f),
            _          => new Color(0.52f, 0.52f, 0.58f)
        };
        environment?.SetPhaseAmbient(col);

        // Adjust orbit speed per phase
        _orbitSpeed  = phase == "random" ? 16f : phase == "training" ? 10f : 8f;
        _orbitHeight = phase == "demo"   ? 1.5f : 1.35f;
        _orbitRadius = phase == "demo"   ? 2.4f : 2.8f;

        _smooth = 1.4f;   // slow drift on phase change
        CamRig rig = phase switch {
            "random"   => RigOverhead,
            "training" => RigTraining,
            "demo"     => RigStoryteller,
            _          => RigStoryteller
        };
        ApplyRig(rig);
        StartCoroutine(RestoreSmooth(2.0f, 0.6f));
    }

    IEnumerator ROSCCameraSequence()
    {
        _smooth = 0.9f; ApplyRig(RigAirway);
        yield return new WaitForSeconds(2.5f);
        _smooth = 1.6f; ApplyRig(RigROSC);
        yield return new WaitForSeconds(3.5f);
        _smooth = 1.0f; ApplyRig(RigStoryteller);
        StartCoroutine(RestoreSmooth(2.0f, 0.6f));
    }

    // ── Camera helpers ────────────────────────────────────────────────────────

    void SetCameraByName(string name)
    {
        ApplyRig(name switch {
            "compression" => RigCompression,
            "airway"      => RigAirway,
            "overhead"    => RigOverhead,
            "rosc"        => RigROSC,
            "training"    => RigTraining,
            _             => RigStoryteller
        });
    }

    void ApplyRig(CamRig rig, bool instant = false)
    {
        _camTgtPos = rig.pos;
        _tgtFov    = rig.fov;
        _camTgtRot = Quaternion.LookRotation(
            (rig.lookAt - rig.pos).normalized, Vector3.up);

        if (instant && mainCamera != null)
        {
            mainCamera.transform.position = rig.pos;
            mainCamera.transform.rotation = _camTgtRot;
            mainCamera.fieldOfView        = rig.fov;
            _camVelPos = Vector3.zero;
            _camVelFov = 0f;
            _orbiting  = false;
            return;
        }

        // Pause orbit for this cut, resume after _smooth + 2s settle time
        _orbiting     = false;
        _orbitResumeT = Time.time + _smooth + 2.0f;
    }

    IEnumerator RestoreSmooth(float delay, float target)
    {
        yield return new WaitForSeconds(delay);
        _smooth = target;
    }
}