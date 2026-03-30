// RescuerController.cs
// =====================
// Controls the rescuer avatar (the person performing CPR).
// Receives action packets and plays corresponding animations.
// Supports both primitive-shape avatars and Humanoid Animator.

using System.Collections;
using UnityEngine;

public class RescuerController : MonoBehaviour
{
    [Header("Rescuer Body Parts (auto-created if null)")]
    public Transform torso;
    public Transform head;
    public Transform leftHand;
    public Transform rightHand;
    public Transform handMarker;   // highlight where hands should be on patient

    [Header("Positioning")]
    public Vector3 kneelPosition  = new Vector3(0.55f, 0.1f, 0.05f);
    public Vector3 standPosition  = new Vector3(0.55f, 0.8f, 0.0f);

    [Header("Humanoid Animator (optional)")]
    public Animator humanoidAnimator;

    // Animator state hashes
    private static readonly int HashCompress    = Animator.StringToHash("Compress");
    private static readonly int HashOpenAirway  = Animator.StringToHash("OpenAirway");
    private static readonly int HashBreath      = Animator.StringToHash("RescueBreath");
    private static readonly int HashAssess      = Animator.StringToHash("Assess");
    private static readonly int HashIdle        = Animator.StringToHash("Idle");

    // ── Action to animation method map ──────────────────────────────────────
    private delegate IEnumerator ActionAnimation();
    private System.Collections.Generic.Dictionary<int, ActionAnimation> _animMap;

    void Awake()
    {
        if (torso == null) BuildPrimitiveRescuer();

        _animMap = new System.Collections.Generic.Dictionary<int, ActionAnimation>
        {
            { 0,  AnimateAssess       },
            { 1,  AnimateCallEmergency},
            { 2,  AnimateOpenAirway   },
            { 3,  AnimateCheckBreath  },
            { 4,  AnimateCompress     },
            { 5,  AnimateRescueBreath },
            { 6,  AnimateDefibrillate },
            { 7,  AnimateMonitorPulse },
            { 8,  AnimateIdle         },
            { 9,  AnimateRepositionHands},
            { 10, AnimateTiltHead     },
            { 11, AnimateIdle         },
        };

        BridgeEvents.OnStateUpdate += HandleState;
    }

    void OnDestroy()
    {
        BridgeEvents.OnStateUpdate -= HandleState;
    }

    private void HandleState(StatePacket p)
    {
        if (_animMap.TryGetValue(p.action, out var anim))
            StartCoroutine(anim());

        // Hand marker: show where hands should be during compressions
        if (handMarker != null)
            handMarker.gameObject.SetActive(p.action == 4 || p.action == 9);
    }

    // ── Action animations ─────────────────────────────────────────────────────

    private IEnumerator AnimateAssess()
    {
        // Rescuer leans over patient, taps shoulders
        yield return MoveHandTo(new Vector3(0f, 0.15f, 0.3f), 0.4f);
        yield return new WaitForSeconds(0.3f);
        yield return MoveHandTo(kneelPosition + Vector3.up * 0.3f, 0.4f);
    }

    private IEnumerator AnimateCallEmergency()
    {
        // Rescuer raises hand (phone gesture)
        yield return MoveHandTo(new Vector3(0.8f, 0.9f, 0f), 0.5f);
        yield return new WaitForSeconds(1.0f);
        yield return MoveHandTo(kneelPosition + Vector3.up * 0.3f, 0.5f);
    }

    private IEnumerator AnimateOpenAirway()
    {
        // Both hands move to patient head area
        yield return MoveHandTo(new Vector3(0f, 0.15f, 0.55f), 0.5f);
        yield return new WaitForSeconds(0.6f);
    }

    private IEnumerator AnimateCheckBreath()
    {
        // Head leans close to patient's mouth
        if (head != null)
        {
            Vector3 orig = head.localPosition;
            head.localPosition = Vector3.Lerp(orig, new Vector3(0f, -0.1f, 0.25f), 0.5f);
            yield return new WaitForSeconds(0.8f);
            head.localPosition = orig;
        }
        yield return null;
    }

    private IEnumerator AnimateCompress()
    {
        // Rapid chest compression: hands pump down repeatedly
        Vector3 chestPos = new Vector3(-0.05f, 0.12f, 0.1f);
        Vector3 upPos    = chestPos + Vector3.up * 0.12f;

        for (int i = 0; i < 6; i++)
        {
            yield return MoveHandTo(chestPos, 0.06f);
            yield return MoveHandTo(upPos,    0.06f);
        }
    }

    private IEnumerator AnimateRescueBreath()
    {
        // Head tilts toward patient's face, breath delivered
        yield return MoveHandTo(new Vector3(0f, 0.15f, 0.52f), 0.4f);
        if (head != null)
        {
            Vector3 orig = head.localPosition;
            head.localPosition = new Vector3(0f, 0.05f, 0.4f);
            yield return new WaitForSeconds(1.0f);
            head.localPosition = orig;
        }
    }

    private IEnumerator AnimateDefibrillate()
    {
        // Rescuer steps to AED, returns with pads
        yield return MoveHandTo(new Vector3(1.2f, 0.25f, 0.5f), 0.8f);
        yield return new WaitForSeconds(0.3f);
        yield return MoveHandTo(new Vector3(-0.1f, 0.15f, 0.1f), 0.8f);
        yield return new WaitForSeconds(0.5f);
        // Stand clear gesture
        if (torso != null)
        {
            torso.localPosition += Vector3.right * 0.5f;
            yield return new WaitForSeconds(0.3f);
            torso.localPosition -= Vector3.right * 0.5f;
        }
    }

    private IEnumerator AnimateMonitorPulse()
    {
        // Hand to patient's neck (carotid)
        yield return MoveHandTo(new Vector3(0.08f, 0.14f, 0.48f), 0.4f);
        yield return new WaitForSeconds(0.8f);
    }

    private IEnumerator AnimateRepositionHands()
    {
        // Hands adjust on chest
        yield return MoveHandTo(new Vector3(0f, 0.14f, 0.15f), 0.3f);
        yield return new WaitForSeconds(0.2f);
    }

    private IEnumerator AnimateTiltHead()
    {
        yield return MoveHandTo(new Vector3(0f, 0.14f, 0.52f), 0.4f);
        yield return new WaitForSeconds(0.5f);
    }

    private IEnumerator AnimateIdle()
    {
        yield return MoveHandTo(kneelPosition + Vector3.up * 0.2f, 0.6f);
    }

    // ── Utility ───────────────────────────────────────────────────────────────

    private IEnumerator MoveHandTo(Vector3 targetLocal, float duration)
    {
        if (leftHand == null && rightHand == null) yield break;

        Transform hand = leftHand ?? rightHand;
        Vector3 start  = hand.localPosition;
        float t = 0f;

        while (t < 1f)
        {
            t = Mathf.MoveTowards(t, 1f, Time.deltaTime / duration);
            hand.localPosition = Vector3.Lerp(start, targetLocal, Mathf.SmoothStep(0, 1, t));
            if (rightHand != null)
                rightHand.localPosition = new Vector3(
                    -hand.localPosition.x + 0.06f,
                     hand.localPosition.y,
                     hand.localPosition.z
                );
            yield return null;
        }
    }

    // ── Primitive rescuer builder ─────────────────────────────────────────────

    private void BuildPrimitiveRescuer()
    {
        Color skinColor = new Color(0.6f, 0.45f, 0.35f);
        Color shirtColor = new Color(0.25f, 0.45f, 0.75f);   // blue shirt

        Transform root = this.transform;
        root.position = kneelPosition;

        torso = CreatePrimitive(PrimitiveType.Cube, root, "R_Torso",
            new Vector3(0.32f, 0.5f, 0.20f), new Vector3(0, 0.25f, 0), shirtColor);

        head = CreatePrimitive(PrimitiveType.Sphere, root, "R_Head",
            new Vector3(0.20f, 0.20f, 0.20f), new Vector3(0, 0.62f, 0), skinColor);

        leftHand = CreatePrimitive(PrimitiveType.Sphere, root, "R_LeftHand",
            new Vector3(0.09f, 0.09f, 0.09f), new Vector3(0.22f, 0.3f, 0.12f), skinColor);

        rightHand = CreatePrimitive(PrimitiveType.Sphere, root, "R_RightHand",
            new Vector3(0.09f, 0.09f, 0.09f), new Vector3(-0.22f, 0.3f, 0.12f), skinColor);

        // Hand placement marker (semi-transparent green ring on patient chest)
        var marker = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        marker.name = "HandMarker";
        marker.transform.position   = new Vector3(0f, 0.06f, 0.1f);
        marker.transform.localScale = new Vector3(0.18f, 0.005f, 0.18f);
        var mr = marker.GetComponent<Renderer>();
        var mat = new Material(Shader.Find("Standard"));
        mat.color = new Color(0.2f, 1f, 0.4f, 0.6f);
        mat.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
        mat.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
        mat.SetInt("_ZWrite", 0);
        mat.EnableKeyword("_ALPHABLEND_ON");
        mat.renderQueue = 3000;
        mr.material = mat;
        handMarker = marker.transform;
        marker.SetActive(false);
    }

    private static Transform CreatePrimitive(
        PrimitiveType type, Transform parent, string name,
        Vector3 scale, Vector3 localPos, Color color)
    {
        var go = GameObject.CreatePrimitive(type);
        go.name = name;
        go.transform.SetParent(parent, false);
        go.transform.localScale    = scale;
        go.transform.localPosition = localPos;
        var r = go.GetComponent<Renderer>();
        if (r != null) { r.material = new Material(Shader.Find("Standard")); r.material.color = color; }
        var col = go.GetComponent<Collider>();
        if (col != null) Object.Destroy(col);
        return go.transform;
    }
}
