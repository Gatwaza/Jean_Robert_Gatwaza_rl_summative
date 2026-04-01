// HumanoidBuilder.cs (v3 — proper kneeling joints, neck close to chest, no crossed legs)
using UnityEngine;

public static class HumanoidBuilder
{
    public static readonly Color SkinTone1   = new Color(0.87f, 0.72f, 0.56f);
    public static readonly Color SkinTone2   = new Color(0.55f, 0.38f, 0.27f);
    public static readonly Color ShirtBlue   = new Color(0.22f, 0.42f, 0.74f);
    public static readonly Color ShirtWhite  = new Color(0.93f, 0.93f, 0.93f);
    public static readonly Color TrouserGrey = new Color(0.30f, 0.30f, 0.35f);
    public static readonly Color ShoeBlack   = new Color(0.12f, 0.12f, 0.14f);
    public static readonly Color HairDark    = new Color(0.18f, 0.12f, 0.08f);

    public class Body
    {
        public Transform root;
        public Transform hips, spine, chest, neck, head;
        public Transform leftShoulder,  leftUpperArm,  leftForearm,  leftWrist,  leftHand;
        public Transform rightShoulder, rightUpperArm, rightForearm, rightWrist, rightHand;
        public Transform leftThigh,  leftShin,  leftFoot;
        public Transform rightThigh, rightShin, rightFoot;
    }

    public static Body Build(Transform root, Color skin, Color shirt, Color trouser, float h = 1f)
    {
        var b = new Body { root = root };

        b.hips  = Node(root,    "Hips",  new Vector3(0, 0.95f*h, 0));
        b.spine = Node(b.hips,  "Spine", new Vector3(0, 0.12f*h, 0));
        b.chest = Node(b.spine, "Chest", new Vector3(0, 0.20f*h, 0));
        // FIX: neck much closer — was 0.26, now 0.12 so no gap between neck/chest
        b.neck  = Node(b.chest, "Neck",  new Vector3(0, 0.12f*h, 0));
        b.head  = Node(b.neck,  "Head",  new Vector3(0, 0.07f*h, 0));

        b.leftShoulder  = Node(b.chest,         "L_Shoulder", new Vector3( 0.18f*h, 0.10f*h, 0));
        b.leftUpperArm  = Node(b.leftShoulder,  "L_UpperArm", new Vector3( 0.13f*h, 0, 0));
        b.leftForearm   = Node(b.leftUpperArm,  "L_Forearm",  new Vector3( 0.25f*h, 0, 0));
        b.leftWrist     = Node(b.leftForearm,   "L_Wrist",    new Vector3( 0.22f*h, 0, 0));
        b.leftHand      = Node(b.leftWrist,     "L_Hand",     new Vector3( 0.04f*h, 0, 0));

        b.rightShoulder = Node(b.chest,         "R_Shoulder", new Vector3(-0.18f*h, 0.10f*h, 0));
        b.rightUpperArm = Node(b.rightShoulder, "R_UpperArm", new Vector3(-0.13f*h, 0, 0));
        b.rightForearm  = Node(b.rightUpperArm, "R_Forearm",  new Vector3(-0.25f*h, 0, 0));
        b.rightWrist    = Node(b.rightForearm,  "R_Wrist",    new Vector3(-0.22f*h, 0, 0));
        b.rightHand     = Node(b.rightWrist,    "R_Hand",     new Vector3(-0.04f*h, 0, 0));

        b.leftThigh  = Node(b.hips, "L_Thigh", new Vector3( 0.11f*h, -0.05f*h, 0));
        b.leftShin   = Node(b.leftThigh,  "L_Shin", new Vector3(0, -0.42f*h, 0));
        b.leftFoot   = Node(b.leftShin,   "L_Foot", new Vector3(0, -0.40f*h, 0));

        b.rightThigh = Node(b.hips, "R_Thigh", new Vector3(-0.11f*h, -0.05f*h, 0));
        b.rightShin  = Node(b.rightThigh, "R_Shin", new Vector3(0, -0.42f*h, 0));
        b.rightFoot  = Node(b.rightShin,  "R_Foot", new Vector3(0, -0.40f*h, 0));

        AttachMeshes(b, skin, shirt, trouser, h);
        return b;
    }

    static void AttachMeshes(Body b, Color skin, Color shirt, Color trouser, float h)
    {
        Seg(b.head, PrimitiveType.Sphere, "HeadMesh",
            new Vector3(0.19f,0.22f,0.19f)*h, new Vector3(0,0.09f*h,0), skin);
        Seg(b.head, PrimitiveType.Sphere, "Hair",
            new Vector3(0.195f,0.13f,0.195f)*h, new Vector3(0,0.16f*h,0), HairDark);
        // Neck mesh — shorter, sits right on chest
        Seg(b.neck, PrimitiveType.Capsule, "NeckMesh",
            new Vector3(0.08f,0.055f,0.08f)*h, new Vector3(0,0.03f*h,0), skin);
        Seg(b.spine, PrimitiveType.Capsule, "TorsoMesh",
            new Vector3(0.28f,0.22f,0.15f)*h, new Vector3(0,0.18f*h,0), shirt);
        Seg(b.hips, PrimitiveType.Capsule, "HipsMesh",
            new Vector3(0.28f,0.10f,0.15f)*h, new Vector3(0,0.05f*h,0), trouser);

        ArmCapsule(b.leftUpperArm,  skin, h, 0.11f, 0.24f,  true);
        ArmCapsule(b.leftForearm,   skin, h, 0.09f, 0.21f,  true);
        Seg(b.leftWrist,  PrimitiveType.Sphere, "L_WristBall",
            new Vector3(0.10f,0.10f,0.10f)*h, Vector3.zero, skin);
        HandPalm(b.leftHand,  skin, h, left:true);

        ArmCapsule(b.rightUpperArm, skin, h, 0.11f, 0.24f, false);
        ArmCapsule(b.rightForearm,  skin, h, 0.09f, 0.21f, false);
        Seg(b.rightWrist, PrimitiveType.Sphere, "R_WristBall",
            new Vector3(0.10f,0.10f,0.10f)*h, Vector3.zero, skin);
        HandPalm(b.rightHand, skin, h, left:false);

        // Shirt cuffs
        Seg(b.leftWrist,  PrimitiveType.Sphere, "L_Cuff",
            new Vector3(0.11f,0.11f,0.11f)*h, Vector3.zero, ShirtWhite);
        Seg(b.rightWrist, PrimitiveType.Sphere, "R_Cuff",
            new Vector3(0.11f,0.11f,0.11f)*h, Vector3.zero, ShirtWhite);

        LegCapsule(b.leftThigh,  trouser, h);
        LegCapsule(b.leftShin,   skin,    h);
        FootMesh(b.leftFoot,  h);
        LegCapsule(b.rightThigh, trouser, h);
        LegCapsule(b.rightShin,  skin,    h);
        FootMesh(b.rightFoot, h);
    }

    static void ArmCapsule(Transform t, Color c, float h, float r, float len, bool left)
    {
        float sign = left ? 1f : -1f;
        var go = GameObject.CreatePrimitive(PrimitiveType.Capsule);
        go.name = "ArmSeg";
        go.transform.SetParent(t, false);
        go.transform.localScale       = new Vector3(r*h, len*h*0.5f, r*h);
        go.transform.localPosition    = new Vector3(sign*len*0.5f*h, 0, 0);
        go.transform.localEulerAngles = new Vector3(0, 0, 90);
        SetColor(go, c); RemoveCollider(go);
    }

    static void HandPalm(Transform hand, Color c, float h, bool left)
    {
        float side = left ? 1f : -1f;
        var palm = GameObject.CreatePrimitive(PrimitiveType.Cube);
        palm.name = "Palm";
        palm.transform.SetParent(hand, false);
        palm.transform.localScale    = new Vector3(0.14f*h, 0.05f*h, 0.11f*h);
        palm.transform.localPosition = new Vector3(side*0.07f*h, 0, 0);
        SetColor(palm, c); RemoveCollider(palm);

        string[] fnames = {"Index","Middle","Ring","Pinky"};
        float[] zOff = { 0.04f, 0.013f, -0.013f, -0.04f };
        float[] lens = { 0.07f, 0.075f,  0.065f,  0.05f };
        for (int i = 0; i < 4; i++)
        {
            var f = GameObject.CreatePrimitive(PrimitiveType.Capsule);
            f.name = fnames[i];
            f.transform.SetParent(hand, false);
            f.transform.localScale       = new Vector3(0.018f*h, lens[i]*h, 0.018f*h);
            f.transform.localPosition    = new Vector3(side*(0.135f+lens[i])*h, 0, zOff[i]*h);
            f.transform.localEulerAngles = new Vector3(0, 0, 90);
            SetColor(f, c); RemoveCollider(f);
        }
        var thumb = GameObject.CreatePrimitive(PrimitiveType.Capsule);
        thumb.name = "Thumb";
        thumb.transform.SetParent(hand, false);
        thumb.transform.localScale       = new Vector3(0.020f*h, 0.045f*h, 0.020f*h);
        thumb.transform.localPosition    = new Vector3(side*0.10f*h, 0, 0.055f*h);
        thumb.transform.localEulerAngles = new Vector3(0, side*(-35f), 90);
        SetColor(thumb, c); RemoveCollider(thumb);
    }

    static void LegCapsule(Transform t, Color c, float h)
    {
        var go = GameObject.CreatePrimitive(PrimitiveType.Capsule);
        go.name = "LegSeg";
        go.transform.SetParent(t, false);
        go.transform.localScale    = new Vector3(0.13f*h, 0.20f*h, 0.13f*h);
        go.transform.localPosition = new Vector3(0, -0.20f*h, 0);
        SetColor(go, c); RemoveCollider(go);
    }

    static void FootMesh(Transform t, float h)
    {
        var go = GameObject.CreatePrimitive(PrimitiveType.Capsule);
        go.name = "FootSeg";
        go.transform.SetParent(t, false);
        go.transform.localScale       = new Vector3(0.09f*h, 0.12f*h, 0.09f*h);
        go.transform.localPosition    = new Vector3(0, -0.05f*h, 0.06f*h);
        go.transform.localEulerAngles = new Vector3(90, 0, 0);
        SetColor(go, ShoeBlack); RemoveCollider(go);
    }

    static void Seg(Transform p, PrimitiveType type, string name,
                    Vector3 scale, Vector3 pos, Color color)
    {
        var go = GameObject.CreatePrimitive(type);
        go.name = name;
        go.transform.SetParent(p, false);
        go.transform.localScale    = scale;
        go.transform.localPosition = pos;
        SetColor(go, color); RemoveCollider(go);
    }

    static Transform Node(Transform parent, string name, Vector3 localPos)
    {
        var go = new GameObject(name);
        go.transform.SetParent(parent, false);
        go.transform.localPosition = localPos;
        return go.transform;
    }

    public static void SetColor(GameObject go, Color c)
    {
        var r = go.GetComponent<Renderer>();
        if (r == null) return;
        var mat = new Material(Shader.Find("Standard"));
        mat.color = c;
        mat.SetFloat("_Glossiness", c == ShoeBlack ? 0.6f : 0.12f);
        r.material = mat;
    }

    static void RemoveCollider(GameObject go)
    {
        var col = go.GetComponent<Collider>();
        if (col != null) Object.Destroy(col);
    }
}