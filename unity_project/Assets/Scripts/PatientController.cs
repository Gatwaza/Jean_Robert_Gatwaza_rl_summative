// PatientController.cs
// =====================
// Drives the patient avatar using MediaPipe landmark data received from Python.
// Supports both a primitive-shape avatar (works out of the box) and
// a Humanoid avatar (attach an Animator with CPR animation clips).
//
// The patient lies supine on the floor. Landmark positions are mapped
// from normalised [0,1] coordinates into Unity world space.
//
// Attach this script to the root of your Patient GameObject.
// Assign body part references in the Inspector.

using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PatientController : MonoBehaviour
{
    [Header("Avatar Mode")]
    [Tooltip("Use primitive shapes (no external assets needed) or Humanoid animator")]
    public bool usePrimitiveAvatar = true;

    [Header("Primitive Body Parts (auto-created if null)")]
    public Transform head;
    public Transform torso;
    public Transform leftArm;
    public Transform rightArm;
    public Transform leftForearm;
    public Transform rightForearm;
    public Transform leftLeg;
    public Transform rightLeg;
    public Transform chest;  // compression highlight zone

    [Header("Humanoid Animator (optional)")]
    public Animator humanoidAnimator;

    [Header("World Space Mapping")]
    [Tooltip("The floor plane Y position in world space")]
    public float floorY = 0f;
    [Tooltip("Scale factor: landmark [0,1] → Unity units")]
    public float worldScale = 2.0f;
    [Tooltip("Centre of the patient in world space")]
    public Vector3 patientOrigin = Vector3.zero;

    [Header("Visual Feedback")]
    public Material correctMaterial;
    public Material incorrectMaterial;
    public Material neutralMaterial;

    // ── Animation state ─────────────────────────────────────────────────────

    private bool  _compressing    = false;
    private float _compressionT   = 0f;
    private bool  _airwayOpen     = false;
    private bool  _recoveryPos    = false;
    private bool  _rosc           = false;
    private Color _heartbeatColor = Color.red;
    private float _heartbeatT     = 0f;

    // ── Animator hashes ─────────────────────────────────────────────────────
    private static readonly int HashCompress  = Animator.StringToHash("Compress");
    private static readonly int HashAirway    = Animator.StringToHash("AirwayOpen");
    private static readonly int HashRecovery  = Animator.StringToHash("Recovery");
    private static readonly int HashROSC      = Animator.StringToHash("ROSC");

    // ── Landmark-to-body-part index map (MediaPipe → body part) ────────────
    // Indices reference the 17-keypoint array from Python
    private const int IDX_NOSE            = 0;
    private const int IDX_LEFT_SHOULDER   = 11;
    private const int IDX_RIGHT_SHOULDER  = 12;
    private const int IDX_LEFT_ELBOW      = 13;
    private const int IDX_RIGHT_ELBOW     = 14;
    private const int IDX_LEFT_WRIST      = 15;
    private const int IDX_RIGHT_WRIST     = 16;
    private const int IDX_LEFT_HIP        = 5;   // approximate (using left eye outer)
    private const int IDX_RIGHT_HIP       = 6;

    void Awake()
    {
        if (usePrimitiveAvatar)
            BuildPrimitiveAvatar();

        // Subscribe to bridge events
        BridgeEvents.OnStateUpdate  += HandleStateUpdate;
        BridgeEvents.OnEpisodeEnd   += HandleEpisodeEnd;
        BridgeEvents.OnPhaseChange  += HandlePhaseChange;
    }

    void OnDestroy()
    {
        BridgeEvents.OnStateUpdate  -= HandleStateUpdate;
        BridgeEvents.OnEpisodeEnd   -= HandleEpisodeEnd;
        BridgeEvents.OnPhaseChange  -= HandlePhaseChange;
    }

    void Update()
    {
        // Chest compression bounce animation
        if (_compressing)
        {
            _compressionT += Time.deltaTime * 8f;  // 120 bpm
            if (chest != null)
            {
                float depth = Mathf.Abs(Mathf.Sin(_compressionT)) * 0.08f;
                chest.localPosition = new Vector3(
                    chest.localPosition.x,
                    -depth,
                    chest.localPosition.z
                );
            }
        }

        // Heartbeat colour pulse (after ROSC)
        if (_rosc && torso != null)
        {
            _heartbeatT += Time.deltaTime * 2f;
            float pulse = 0.5f + 0.5f * Mathf.Sin(_heartbeatT * Mathf.PI * 2);
            Color col = Color.Lerp(Color.white, new Color(1f, 0.3f, 0.3f), pulse);
            SetBodyColor(torso, col);
        }
    }

    // ── Event handlers ───────────────────────────────────────────────────────

    private void HandleStateUpdate(StatePacket p)
    {
        if (p.landmarks != null && p.landmarks.Length >= 51)
            ApplyLandmarks(p.landmarks);

        // Update animation flags
        _compressing = (p.action == 4);
        _airwayOpen  = p.vitals?.airway_open ?? false;
        _recoveryPos = p.vitals?.recovery_position ?? false;
        _rosc        = p.rosc;

        if (!_compressing)
        {
            _compressionT = 0f;
            if (chest != null)
                chest.localPosition = new Vector3(chest.localPosition.x, 0f, chest.localPosition.z);
        }

        // Head tilt for airway actions
        if (p.action == 2 || p.action == 10)
            AnimateHeadTilt(_airwayOpen ? 25f : 5f);

        // Recovery position
        if (_recoveryPos)
            AnimateRecoveryPosition();

        // Drive Humanoid Animator if available
        if (humanoidAnimator != null)
        {
            humanoidAnimator.SetBool(HashCompress, _compressing);
            humanoidAnimator.SetBool(HashAirway,   _airwayOpen);
            humanoidAnimator.SetBool(HashRecovery, _recoveryPos);
            humanoidAnimator.SetBool(HashROSC,     _rosc);
        }

        // Visual feedback colour
        ApplyActionFeedback(p.is_correct, p.action);
    }

    private void HandleEpisodeEnd(StatePacket p)
    {
        _compressing = false;
        if (p.rosc)
        {
            // Sit up animation on ROSC
            StartCoroutine(AnimateROSC());
        }
        else
        {
            // Reset to supine
            StartCoroutine(ResetPose());
        }
    }

    private void HandlePhaseChange(StatePacket p)
    {
        // Reset patient to collapsed state on new phase/experiment
        StartCoroutine(ResetPose());
    }

    // ── Landmark application ─────────────────────────────────────────────────

    private void ApplyLandmarks(float[] lm)
    {
        // Helper: get world position from landmark index
        Vector3 LandmarkPos(int idx)
        {
            float nx = lm[idx * 3];       // normalised x [0,1]
            float ny = lm[idx * 3 + 1];   // normalised y [0,1] (0=top, 1=bottom)
            // Map to world space: patient lying flat, y → z (depth), ny → height offset
            return patientOrigin + new Vector3(
                (nx - 0.5f) * worldScale,
                floorY + (1f - ny) * worldScale * 0.4f,  // flat body
                (ny - 0.85f) * worldScale * 2f            // along body length
            );
        }

        // Position body parts using landmark data
        SafeSetPosition(head,         LandmarkPos(IDX_NOSE));
        SafeSetPosition(leftArm,      LandmarkPos(IDX_LEFT_SHOULDER));
        SafeSetPosition(rightArm,     LandmarkPos(IDX_RIGHT_SHOULDER));
        SafeSetPosition(leftForearm,  LandmarkPos(IDX_LEFT_ELBOW));
        SafeSetPosition(rightForearm, LandmarkPos(IDX_RIGHT_ELBOW));

        // Torso midpoint between shoulders
        if (torso != null && lm.Length >= 39)
        {
            Vector3 ls = LandmarkPos(IDX_LEFT_SHOULDER);
            Vector3 rs = LandmarkPos(IDX_RIGHT_SHOULDER);
            torso.position = Vector3.Lerp(ls, rs, 0.5f);
        }
    }

    private static void SafeSetPosition(Transform t, Vector3 pos)
    {
        if (t != null)
            t.position = Vector3.Lerp(t.position, pos, Time.deltaTime * 12f);
    }

    // ── Procedural animations ────────────────────────────────────────────────

    private void AnimateHeadTilt(float angle)
    {
        if (head == null) return;
        Vector3 target = head.localEulerAngles;
        target.x = angle;
        head.localEulerAngles = Vector3.Lerp(
            head.localEulerAngles, target, Time.deltaTime * 5f
        );
    }

    private void AnimateRecoveryPosition()
    {
        if (torso == null) return;
        Quaternion target = Quaternion.Euler(0f, 0f, 70f);  // side roll
        torso.localRotation = Quaternion.Slerp(
            torso.localRotation, target, Time.deltaTime * 2f
        );
    }

    private IEnumerator AnimateROSC()
    {
        _rosc = true;
        // Patient shows signs of life: subtle arm movement, eye flutter
        float t = 0f;
        while (t < 3f)
        {
            t += Time.deltaTime;
            if (head != null)
            {
                float nod = Mathf.Sin(t * 2f) * 3f;
                head.localEulerAngles = new Vector3(nod, 0f, 0f);
            }
            yield return null;
        }
    }

    private IEnumerator ResetPose()
    {
        _rosc        = false;
        _compressing = false;
        _airwayOpen  = false;
        _recoveryPos = false;

        float t = 0f;
        while (t < 0.5f)
        {
            t += Time.deltaTime;
            if (head  != null) head.localRotation  = Quaternion.Slerp(head.localRotation,  Quaternion.identity, t * 2f);
            if (torso != null) torso.localRotation = Quaternion.Slerp(torso.localRotation, Quaternion.identity, t * 2f);
            if (torso != null) SetBodyColor(torso, Color.white);
            yield return null;
        }
    }

    // ── Visual feedback ──────────────────────────────────────────────────────

    private void ApplyActionFeedback(bool isCorrect, int action)
    {
        if (torso == null) return;

        if (_rosc)
            return;  // heartbeat colour handled in Update

        Material mat = isCorrect ? correctMaterial : incorrectMaterial;
        if (mat != null)
        {
            var renderer = torso.GetComponent<Renderer>();
            if (renderer != null)
            {
                renderer.material = mat;
                StartCoroutine(RevertMaterialAfterDelay(renderer, 0.5f));
            }
        }
    }

    private IEnumerator RevertMaterialAfterDelay(Renderer r, float delay)
    {
        yield return new WaitForSeconds(delay);
        if (r != null && neutralMaterial != null)
            r.material = neutralMaterial;
    }

    private static void SetBodyColor(Transform t, Color c)
    {
        var r = t.GetComponent<Renderer>();
        if (r != null) r.material.color = c;
    }

    // ── Primitive avatar builder ─────────────────────────────────────────────

    private void BuildPrimitiveAvatar()
    {
        // Creates a simple humanoid from Unity primitive shapes.
        // Patient lies face-up on the floor.

        Transform root = this.transform;

        torso        = CreatePrimitive(PrimitiveType.Cube,     root, "Torso",
                           new Vector3(0.4f, 0.08f, 0.7f),
                           new Vector3(0, 0.04f, 0),
                           new Color(0.87f, 0.72f, 0.56f));

        head         = CreatePrimitive(PrimitiveType.Sphere,   root, "Head",
                           new Vector3(0.22f, 0.22f, 0.22f),
                           new Vector3(0, 0.04f, 0.52f),
                           new Color(0.87f, 0.72f, 0.56f));

        chest        = CreatePrimitive(PrimitiveType.Cube,     torso, "Chest",
                           new Vector3(0.35f, 0.06f, 0.35f),
                           new Vector3(0, 0.07f, 0.1f),
                           new Color(0.85f, 0.70f, 0.54f));

        leftArm      = CreatePrimitive(PrimitiveType.Capsule,  root, "LeftArm",
                           new Vector3(0.1f, 0.22f, 0.1f),
                           new Vector3(0.35f, 0.04f, 0.25f),
                           new Color(0.87f, 0.72f, 0.56f));

        rightArm     = CreatePrimitive(PrimitiveType.Capsule,  root, "RightArm",
                           new Vector3(0.1f, 0.22f, 0.1f),
                           new Vector3(-0.35f, 0.04f, 0.25f),
                           new Color(0.87f, 0.72f, 0.56f));

        leftForearm  = CreatePrimitive(PrimitiveType.Capsule,  root, "LeftForearm",
                           new Vector3(0.08f, 0.18f, 0.08f),
                           new Vector3(0.38f, 0.04f, -0.05f),
                           new Color(0.87f, 0.72f, 0.56f));

        rightForearm = CreatePrimitive(PrimitiveType.Capsule,  root, "RightForearm",
                           new Vector3(0.08f, 0.18f, 0.08f),
                           new Vector3(-0.38f, 0.04f, -0.05f),
                           new Color(0.87f, 0.72f, 0.56f));

        leftLeg      = CreatePrimitive(PrimitiveType.Capsule,  root, "LeftLeg",
                           new Vector3(0.12f, 0.35f, 0.12f),
                           new Vector3(0.15f, 0.04f, -0.45f),
                           new Color(0.87f, 0.72f, 0.56f));

        rightLeg     = CreatePrimitive(PrimitiveType.Capsule,  root, "RightLeg",
                           new Vector3(0.12f, 0.35f, 0.12f),
                           new Vector3(-0.15f, 0.04f, -0.45f),
                           new Color(0.87f, 0.72f, 0.56f));

        // Rotate capsules to lie flat
        RotateLying(leftArm);  RotateLying(rightArm);
        RotateLying(leftForearm);  RotateLying(rightForearm);
        RotateLying(leftLeg);  RotateLying(rightLeg);
    }

    private static Transform CreatePrimitive(
        PrimitiveType type, Transform parent, string name,
        Vector3 scale, Vector3 localPos, Color color)
    {
        GameObject go = GameObject.CreatePrimitive(type);
        go.name = name;
        go.transform.SetParent(parent, false);
        go.transform.localScale    = scale;
        go.transform.localPosition = localPos;
        var r = go.GetComponent<Renderer>();
        if (r != null)
        {
            r.material = new Material(Shader.Find("Standard"));
            r.material.color = color;
        }
        // Remove collider (visual only)
        var col = go.GetComponent<Collider>();
        if (col != null) Destroy(col);
        return go.transform;
    }

    private static void RotateLying(Transform t)
    {
        if (t != null)
            t.localEulerAngles = new Vector3(0f, 0f, 90f);
    }
}
