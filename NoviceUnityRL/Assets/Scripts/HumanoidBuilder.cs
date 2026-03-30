// HumanoidBuilder.cs
// ====================
// Procedurally constructs an anatomically-proportioned humanoid avatar
// from Unity primitive shapes. No external assets required.
//
// Joint hierarchy mirrors a standard Humanoid rig so Animation Rigging
// IK constraints (Two Bone IK, Multi-Aim) can be applied on top.
//
// Body proportions follow the "8-head" artistic standard:
//   Total height = 8 × head height.
//   Used by both HumanoidPatient and HumanoidRescuer.

using System.Collections.Generic;
using UnityEngine;

public static class HumanoidBuilder
{
    // ── Skin / clothing colour presets ──────────────────────────────────────
    public static readonly Color SkinTone1  = new Color(0.87f, 0.72f, 0.56f);
    public static readonly Color SkinTone2  = new Color(0.55f, 0.38f, 0.27f);
    public static readonly Color ShirtBlue  = new Color(0.22f, 0.42f, 0.74f);
    public static readonly Color ShirtWhite = new Color(0.93f, 0.93f, 0.93f);
    public static readonly Color TrouserGrey= new Color(0.30f, 0.30f, 0.35f);
    public static readonly Color ShoeBlack  = new Color(0.12f, 0.12f, 0.14f);
    public static readonly Color HairDark   = new Color(0.18f, 0.12f, 0.08f);

    // ── Joint reference struct ───────────────────────────────────────────────
    public class Body
    {
        public Transform root;
        public Transform hips;
        public Transform spine, chest, neck, head;
        public Transform leftShoulder, leftUpperArm, leftForearm, leftHand;
        public Transform rightShoulder, rightUpperArm, rightForearm, rightHand;
        public Transform leftThigh, leftShin, leftFoot;
        public Transform rightThigh, rightShin, rightFoot;
    }

    // ── Main entry point ─────────────────────────────────────────────────────
    /// <summary>
    /// Build a complete humanoid under <paramref name="root"/>.
    /// Returns a Body struct with all joint transforms.
    /// </summary>
    public static Body Build(
        Transform root,
        Color skinColor,
        Color shirtColor,
        Color trouserColor,
        float heightScale = 1f)
    {
        float h = heightScale;
        var body = new Body { root = root };

        // ── Hips (pelvis — world anchor) ─────────────────────────────────────
        body.hips = Node(root, "Hips",
            new Vector3(0, 0.95f * h, 0));

        // ── Spine chain ──────────────────────────────────────────────────────
        body.spine = Node(body.hips, "Spine",
            new Vector3(0, 0.12f * h, 0));
        body.chest = Node(body.spine, "Chest",
            new Vector3(0, 0.20f * h, 0));
        body.neck  = Node(body.chest, "Neck",
            new Vector3(0, 0.26f * h, 0));
        body.head  = Node(body.neck, "Head",
            new Vector3(0, 0.08f * h, 0));

        // ── Left arm ─────────────────────────────────────────────────────────
        body.leftShoulder   = Node(body.chest, "L_Shoulder",
            new Vector3( 0.18f * h, 0.22f * h, 0));
        body.leftUpperArm   = Node(body.leftShoulder, "L_UpperArm",
            new Vector3( 0.12f * h, 0, 0));
        body.leftForearm    = Node(body.leftUpperArm, "L_Forearm",
            new Vector3( 0.25f * h, 0, 0));
        body.leftHand       = Node(body.leftForearm, "L_Hand",
            new Vector3( 0.22f * h, 0, 0));

        // ── Right arm ────────────────────────────────────────────────────────
        body.rightShoulder  = Node(body.chest, "R_Shoulder",
            new Vector3(-0.18f * h, 0.22f * h, 0));
        body.rightUpperArm  = Node(body.rightShoulder, "R_UpperArm",
            new Vector3(-0.12f * h, 0, 0));
        body.rightForearm   = Node(body.rightUpperArm, "R_Forearm",
            new Vector3(-0.25f * h, 0, 0));
        body.rightHand      = Node(body.rightForearm, "R_Hand",
            new Vector3(-0.22f * h, 0, 0));

        // ── Left leg ─────────────────────────────────────────────────────────
        body.leftThigh  = Node(body.hips, "L_Thigh",
            new Vector3( 0.10f * h, -0.05f * h, 0));
        body.leftShin   = Node(body.leftThigh, "L_Shin",
            new Vector3(0, -0.42f * h, 0));
        body.leftFoot   = Node(body.leftShin, "L_Foot",
            new Vector3(0, -0.40f * h, 0));

        // ── Right leg ────────────────────────────────────────────────────────
        body.rightThigh = Node(body.hips, "R_Thigh",
            new Vector3(-0.10f * h, -0.05f * h, 0));
        body.rightShin  = Node(body.rightThigh, "R_Shin",
            new Vector3(0, -0.42f * h, 0));
        body.rightFoot  = Node(body.rightShin, "R_Foot",
            new Vector3(0, -0.40f * h, 0));

        // ── Attach mesh segments ─────────────────────────────────────────────
        AttachMeshes(body, skinColor, shirtColor, trouserColor, h);

        return body;
    }

    // ── Mesh attachment ───────────────────────────────────────────────────────
    static void AttachMeshes(Body b, Color skin, Color shirt, Color trouser, float h)
    {
        // Head
        Seg(b.head, PrimitiveType.Sphere, "HeadMesh",
            new Vector3(0.19f, 0.22f, 0.19f) * h,
            new Vector3(0, 0.09f * h, 0), skin);

        // Hair cap
        Seg(b.head, PrimitiveType.Sphere, "Hair",
            new Vector3(0.195f, 0.14f, 0.195f) * h,
            new Vector3(0, 0.16f * h, 0), HairDark);

        // Neck
        Seg(b.neck, PrimitiveType.Capsule, "NeckMesh",
            new Vector3(0.08f, 0.06f, 0.08f) * h,
            Vector3.zero, skin);

        // Torso — shirt
        Seg(b.spine, PrimitiveType.Capsule, "TorsoMesh",
            new Vector3(0.28f, 0.22f, 0.15f) * h,
            new Vector3(0, 0.18f * h, 0), shirt);

        // Pelvis
        Seg(b.hips, PrimitiveType.Capsule, "HipsMesh",
            new Vector3(0.28f, 0.10f, 0.15f) * h,
            new Vector3(0, 0.05f * h, 0), trouser);

        // Arms
        ArmSeg(b.leftUpperArm,  skin,   h, 0.11f, 0.24f, true);
        ArmSeg(b.leftForearm,   skin,   h, 0.09f, 0.22f, true);
        HandSeg(b.leftHand,     skin,   h);
        ArmSeg(b.rightUpperArm, skin,   h, 0.11f, 0.24f, false);
        ArmSeg(b.rightForearm,  skin,   h, 0.09f, 0.22f, false);
        HandSeg(b.rightHand,    skin,   h);

        // Shirt sleeve cuffs
        Seg(b.leftForearm,  PrimitiveType.Capsule, "LeftCuff",
            new Vector3(0.09f, 0.05f, 0.09f) * h,
            new Vector3(0.20f * h, 0, 0), ShirtWhite);
        Seg(b.rightForearm, PrimitiveType.Capsule, "RightCuff",
            new Vector3(0.09f, 0.05f, 0.09f) * h,
            new Vector3(-0.20f * h, 0, 0), ShirtWhite);

        // Legs
        LegSeg(b.leftThigh,  trouser, h, true);
        LegSeg(b.leftShin,   trouser, h, true);
        FootSeg(b.leftFoot,  h);
        LegSeg(b.rightThigh, trouser, h, false);
        LegSeg(b.rightShin,  trouser, h, false);
        FootSeg(b.rightFoot, h);
    }

    static void ArmSeg(Transform t, Color c, float h, float r, float len, bool left)
    {
        float sign = left ? 1f : -1f;
        var go = GameObject.CreatePrimitive(PrimitiveType.Capsule);
        go.name = "ArmSeg";
        go.transform.SetParent(t, false);
        go.transform.localScale    = new Vector3(r * h, len * h * 0.5f, r * h);
        go.transform.localPosition = new Vector3(sign * len * 0.5f * h, 0, 0);
        go.transform.localEulerAngles = new Vector3(0, 0, 90);
        SetColor(go, c); RemoveCollider(go);
    }

    static void HandSeg(Transform t, Color c, float h)
    {
        var go = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        go.name = "HandSeg";
        go.transform.SetParent(t, false);
        go.transform.localScale    = new Vector3(0.09f, 0.06f, 0.12f) * h;
        go.transform.localPosition = Vector3.zero;
        SetColor(go, c); RemoveCollider(go);

        // Fingers — 4 small capsules
        for (int i = 0; i < 4; i++)
        {
            float xOff = (i - 1.5f) * 0.025f * h;
            var f = GameObject.CreatePrimitive(PrimitiveType.Capsule);
            f.name = "Finger" + i;
            f.transform.SetParent(t, false);
            f.transform.localScale    = new Vector3(0.018f, 0.035f, 0.018f) * h;
            f.transform.localPosition = new Vector3(xOff, 0, 0.07f * h);
            SetColor(f, c); RemoveCollider(f);
        }
        // Thumb
        var th = GameObject.CreatePrimitive(PrimitiveType.Capsule);
        th.name = "Thumb";
        th.transform.SetParent(t, false);
        th.transform.localScale    = new Vector3(0.022f, 0.028f, 0.022f) * h;
        th.transform.localPosition = new Vector3(0.065f * h, 0, 0.04f * h);
        th.transform.localEulerAngles = new Vector3(0, 0, -40);
        SetColor(th, c); RemoveCollider(th);
    }

    static void LegSeg(Transform t, Color c, float h, bool left)
    {
        var go = GameObject.CreatePrimitive(PrimitiveType.Capsule);
        go.name = "LegSeg";
        go.transform.SetParent(t, false);
        go.transform.localScale    = new Vector3(0.13f, 0.20f, 0.13f) * h;
        go.transform.localPosition = new Vector3(0, -0.20f * h, 0);
        SetColor(go, c); RemoveCollider(go);
    }

    static void FootSeg(Transform t, float h)
    {
        var go = GameObject.CreatePrimitive(PrimitiveType.Capsule);
        go.name = "FootSeg";
        go.transform.SetParent(t, false);
        go.transform.localScale    = new Vector3(0.09f, 0.12f, 0.09f) * h;
        go.transform.localPosition = new Vector3(0, -0.05f * h, 0.06f * h);
        go.transform.localEulerAngles = new Vector3(90, 0, 0);
        SetColor(go, ShoeBlack); RemoveCollider(go);
    }

    static void Seg(Transform parent, PrimitiveType type, string name,
                    Vector3 scale, Vector3 localPos, Color color)
    {
        var go = GameObject.CreatePrimitive(type);
        go.name = name;
        go.transform.SetParent(parent, false);
        go.transform.localScale    = scale;
        go.transform.localPosition = localPos;
        SetColor(go, color); RemoveCollider(go);
    }

    // ── Helpers ───────────────────────────────────────────────────────────────
    static Transform Node(Transform parent, string name, Vector3 localPos)
    {
        var go = new GameObject(name);
        go.transform.SetParent(parent, false);
        go.transform.localPosition = localPos;
        return go.transform;
    }

    static void SetColor(GameObject go, Color c)
    {
        var r = go.GetComponent<Renderer>();
        if (r == null) return;
        var mat = new Material(Shader.Find("Standard"));
        mat.color = c;
        // Slight roughness for skin/cloth feel
        mat.SetFloat("_Glossiness", c == ShoeBlack ? 0.6f : 0.15f);
        r.material = mat;
    }

    static void RemoveCollider(GameObject go)
    {
        var col = go.GetComponent<Collider>();
        if (col != null) Object.Destroy(col);
    }
}
