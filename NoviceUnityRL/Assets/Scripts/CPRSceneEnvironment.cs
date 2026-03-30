// CPRSceneEnvironment.cs
// ========================
// Procedurally builds the full 3D scene environment at runtime.
// No external assets required.
//
// Scene contents:
//   - Tiled floor (dark clinical linoleum look)
//   - Four walls with subtle baseboard
//   - Red Cross feature wall (back wall) with raised emblem
//   - Dropped ceiling with recessed light panels
//   - Floor mat (blue CPR training pad) under patient
//   - AED device on wheeled stand (right wall)
//   - IV drip stand prop (background)
//   - Dynamic point lights for atmosphere
//   - Spot light above patient (surgical lamp style)
//
// Attach to an empty "Environment" GameObject.
// Call from CPRSceneManager after Start().

using System.Collections;
using UnityEngine;

public class CPRSceneEnvironment : MonoBehaviour
{
    [Header("Scene Dimensions")]
    public float roomWidth  = 8f;
    public float roomDepth  = 6f;
    public float roomHeight = 3.2f;

    [Header("Lights")]
    public Light surgicalSpot;      // assigned by this script, exposed for intensity
    public Light ambientFill;

    // ── Internal refs ────────────────────────────────────────────────────────
    private GameObject _redCrossWall;
    private GameObject _aedStand;
    private Light      _surgSpot;

    // Expose AED position for rescuer IK target
    public Vector3 AEDWorldPosition =>
        _aedStand != null
            ? _aedStand.transform.position + Vector3.up * 0.5f
            : Vector3.right * 2f + Vector3.up * 0.5f;

    void Start() => BuildScene();

    // ── Public phase API ──────────────────────────────────────────────────────
    public void SetPhaseAmbient(Color target, float duration = 1.5f)
    {
        StartCoroutine(FadeAmbient(target, duration));
    }

    public void FlashROSC()    { StartCoroutine(ROSCFlash()); }
    public void FlashIncorrect() { StartCoroutine(IncorrectFlash()); }

    // ── Scene builder ─────────────────────────────────────────────────────────
    void BuildScene()
    {
        BuildFloor();
        BuildCeiling();
        BuildWalls();
        BuildRedCrossWall();
        BuildFloorMat();
        BuildAED();
        BuildIVStand();
        BuildLights();
    }

    void BuildFloor()
    {
        // Dark grey linoleum tiles
        var floor = GameObject.CreatePrimitive(PrimitiveType.Plane);
        floor.name = "Floor";
        floor.transform.SetParent(transform);
        floor.transform.localScale = new Vector3(roomWidth / 10f, 1, roomDepth / 10f);
        floor.transform.position   = Vector3.zero;
        SetMaterial(floor, new Color(0.16f, 0.17f, 0.18f), 0.5f);
    }

    void BuildCeiling()
    {
        var ceil = GameObject.CreatePrimitive(PrimitiveType.Plane);
        ceil.name = "Ceiling";
        ceil.transform.SetParent(transform);
        ceil.transform.localScale    = new Vector3(roomWidth / 10f, 1, roomDepth / 10f);
        ceil.transform.position      = new Vector3(0, roomHeight, 0);
        ceil.transform.localRotation = Quaternion.Euler(180, 0, 0);
        SetMaterial(ceil, new Color(0.92f, 0.92f, 0.90f), 0.1f);

        // Two recessed light panels
        for (int i = -1; i <= 1; i += 2)
        {
            var panel = GameObject.CreatePrimitive(PrimitiveType.Cube);
            panel.name = "LightPanel" + i;
            panel.transform.SetParent(transform);
            panel.transform.localScale = new Vector3(1.2f, 0.04f, 0.4f);
            panel.transform.position   = new Vector3(i * 1.8f, roomHeight - 0.02f, 0);
            SetEmissive(panel, new Color(1f, 0.98f, 0.92f), 1.2f);
        }
    }

    void BuildWalls()
    {
        Color wallColor = new Color(0.88f, 0.87f, 0.84f);
        float h = roomHeight;
        float w = roomWidth;
        float d = roomDepth;

        // Side walls replaced with open half-walls for depth framing
        // (full walls blocked camera — replaced with low skirting panels)
        HalfWall("PanelLeft",  new Vector3(-w/2, 0.4f, 0), d);
        HalfWall("PanelRight", new Vector3( w/2, 0.4f, 0), d);
    }


    // Low decorative skirting panel — gives depth without blocking camera
    void HalfWall(string name, Vector3 pos, float length)
    {
        // Main panel
        var go = GameObject.CreatePrimitive(PrimitiveType.Cube);
        go.name = name;
        go.transform.SetParent(transform);
        go.transform.localScale = new Vector3(0.12f, 0.8f, length);
        go.transform.position   = pos;
        SetMaterial(go, new Color(0.84f, 0.83f, 0.80f), 0.08f);
        var col = go.GetComponent<Collider>(); if (col) Destroy(col);

        // Cap strip on top of panel
        var cap = GameObject.CreatePrimitive(PrimitiveType.Cube);
        cap.name = name + "_Cap";
        cap.transform.SetParent(transform);
        cap.transform.localScale = new Vector3(0.16f, 0.04f, length);
        cap.transform.position   = pos + Vector3.up * 0.42f;
        SetMaterial(cap, new Color(0.70f, 0.68f, 0.65f), 0.25f);
        var col2 = cap.GetComponent<Collider>(); if (col2) Destroy(col2);
    }

    void Wall(string name, Vector3 pos, Vector3 scale, Color color, float rotY)
    {
        var go = GameObject.CreatePrimitive(PrimitiveType.Cube);
        go.name = name;
        go.transform.SetParent(transform);
        go.transform.localScale    = scale;
        go.transform.position      = pos;
        go.transform.localRotation = Quaternion.Euler(0, rotY, 0);
        SetMaterial(go, color, 0.05f);
        var col = go.GetComponent<Collider>();
        if (col) Destroy(col);
    }

    void Baseboard(string name, Vector3 pos, float length)
    {
        var go = GameObject.CreatePrimitive(PrimitiveType.Cube);
        go.name = name;
        go.transform.SetParent(transform);
        go.transform.localScale = new Vector3(0.05f, 0.12f, length);
        go.transform.position   = pos;
        SetMaterial(go, new Color(0.75f, 0.75f, 0.73f), 0.15f);
        var col = go.GetComponent<Collider>();
        if (col) Destroy(col);
    }

    // ── Red Cross feature wall ────────────────────────────────────────────────
    void BuildRedCrossWall()
    {
        float h = roomHeight;
        float w = roomWidth;

        // Main wall — off-white with warm tint
        var wall = GameObject.CreatePrimitive(PrimitiveType.Cube);
        wall.name = "WallBack_RedCross";
        wall.transform.SetParent(transform);
        wall.transform.localScale = new Vector3(w, h, 0.15f);
        wall.transform.position   = new Vector3(0, h / 2, -roomDepth / 2);
        SetMaterial(wall, new Color(0.95f, 0.94f, 0.92f), 0.05f);
        var col = wall.GetComponent<Collider>();
        if (col) Destroy(col);
        _redCrossWall = wall;

        // ── Red Cross emblem ──────────────────────────────────────────────────
        // The emblem is a raised 3D cross — two rectangular boxes
        // arranged in a plus shape, painted in exact Red Cross red.
        Color rcRed   = new Color(0.822f, 0.063f, 0.063f);  // #D11010 — Red Cross red
        Color rcWhite = new Color(1f, 1f, 1f);
        Vector3 center = new Vector3(0, h * 0.62f, -roomDepth / 2 + 0.08f);
        float  emblemScale = 0.90f;

        // White square backing plate (circle approximation with square)
        var backing = GameObject.CreatePrimitive(PrimitiveType.Cube);
        backing.name = "CrossBacking";
        backing.transform.SetParent(transform);
        backing.transform.localScale = new Vector3(0.74f, 0.74f, 0.03f) * emblemScale;
        backing.transform.position   = center - Vector3.forward * 0.015f;
        SetMaterial(backing, rcWhite, 0.1f);
        RemoveCollider(backing);

        // Red rectangle — vertical bar
        var vBar = GameObject.CreatePrimitive(PrimitiveType.Cube);
        vBar.name = "CrossVertical";
        vBar.transform.SetParent(transform);
        vBar.transform.localScale = new Vector3(0.22f, 0.62f, 0.04f) * emblemScale;
        vBar.transform.position   = center;
        SetMaterial(vBar, rcRed, 0.3f);
        RemoveCollider(vBar);

        // Red rectangle — horizontal bar
        var hBar = GameObject.CreatePrimitive(PrimitiveType.Cube);
        hBar.name = "CrossHorizontal";
        hBar.transform.SetParent(transform);
        hBar.transform.localScale = new Vector3(0.62f, 0.22f, 0.04f) * emblemScale;
        hBar.transform.position   = center;
        SetMaterial(hBar, rcRed, 0.3f);
        RemoveCollider(hBar);

        // "CROIX ROUGE" lettering approximation — two thin text-bar blocks
        // (representing lettering without needing a font asset)
        var textBar = GameObject.CreatePrimitive(PrimitiveType.Cube);
        textBar.name = "CroixRougeBar";
        textBar.transform.SetParent(transform);
        textBar.transform.localScale = new Vector3(0.65f, 0.055f, 0.025f) * emblemScale;
        textBar.transform.position   = center + Vector3.down * 0.44f * emblemScale;
        SetMaterial(textBar, rcRed, 0.2f);
        RemoveCollider(textBar);

        // Subtle point light illuminating the emblem from below
        var lightGo = new GameObject("CrossLight");
        lightGo.transform.SetParent(transform);
        lightGo.transform.position = center + Vector3.down * 0.6f + Vector3.forward * 0.4f;
        var pt = lightGo.AddComponent<Light>();
        pt.type      = LightType.Point;
        pt.color     = new Color(1f, 0.9f, 0.88f);
        pt.intensity = 0.6f;
        pt.range     = 2.0f;
    }

    // ── Floor mat (CPR training pad) ──────────────────────────────────────────
    void BuildFloorMat()
    {
        // Outer mat — dark blue
        var mat = GameObject.CreatePrimitive(PrimitiveType.Cube);
        mat.name = "FloorMat";
        mat.transform.SetParent(transform);
        mat.transform.localScale = new Vector3(1.0f, 0.025f, 2.2f);
        mat.transform.position   = new Vector3(0, 0.012f, 0);
        SetMaterial(mat, new Color(0.10f, 0.16f, 0.40f), 0.05f);
        RemoveCollider(mat);

        // Centre guide stripe — lighter blue
        var stripe = GameObject.CreatePrimitive(PrimitiveType.Cube);
        stripe.name = "MatStripe";
        stripe.transform.SetParent(transform);
        stripe.transform.localScale = new Vector3(0.08f, 0.026f, 2.0f);
        stripe.transform.position   = new Vector3(0, 0.013f, 0);
        SetMaterial(stripe, new Color(0.25f, 0.45f, 0.85f), 0.05f);
        RemoveCollider(stripe);

        // Hand position markers (two small circles = squares showing where to place hands)
        for (int i = 0; i < 2; i++)
        {
            var marker = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            marker.name = "HandMarker" + i;
            marker.transform.SetParent(transform);
            marker.transform.localScale = new Vector3(0.12f, 0.007f, 0.12f);
            marker.transform.position   = new Vector3((i - 0.5f) * 0.12f, 0.027f, 0.08f);
            SetMaterial(marker, new Color(1f, 0.85f, 0.1f), 0.4f);  // yellow guide markers
            RemoveCollider(marker);
        }
    }

    // ── AED device on stand ───────────────────────────────────────────────────
    void BuildAED()
    {
        float x = roomWidth / 2 - 0.5f;

        // Stand pole
        var pole = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        pole.name = "AEDPole";
        pole.transform.SetParent(transform);
        pole.transform.localScale = new Vector3(0.04f, 0.55f, 0.04f);
        pole.transform.position   = new Vector3(x, 0.55f, 0.5f);
        SetMaterial(pole, new Color(0.6f, 0.6f, 0.65f), 0.7f);
        RemoveCollider(pole);

        // Stand base
        var standBase = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        standBase.name = "AEDBase";
        standBase.transform.SetParent(transform);
        standBase.transform.localScale = new Vector3(0.28f, 0.025f, 0.28f);
        standBase.transform.position   = new Vector3(x, 0.012f, 0.5f);
        SetMaterial(standBase, new Color(0.4f, 0.4f, 0.45f), 0.6f);
        RemoveCollider(standBase);

        // AED device body — bright yellow-orange (defibrillator color)
        var aed = GameObject.CreatePrimitive(PrimitiveType.Cube);
        aed.name = "AEDDevice";
        aed.transform.SetParent(transform);
        aed.transform.localScale = new Vector3(0.26f, 0.18f, 0.10f);
        aed.transform.position   = new Vector3(x, 0.98f, 0.5f);
        SetMaterial(aed, new Color(1f, 0.52f, 0.05f), 0.35f);
        RemoveCollider(aed);

        // Red cross on AED face
        var cross = GameObject.CreatePrimitive(PrimitiveType.Cube);
        cross.name = "AEDCross";
        cross.transform.SetParent(transform);
        cross.transform.localScale = new Vector3(0.06f, 0.10f, 0.015f);
        cross.transform.position   = new Vector3(x - 0.06f, 0.98f, 0.45f);
        SetMaterial(cross, new Color(0.85f, 0.05f, 0.05f), 0.2f);
        RemoveCollider(cross);

        // Screen panel
        var screen = GameObject.CreatePrimitive(PrimitiveType.Cube);
        screen.name = "AEDScreen";
        screen.transform.SetParent(transform);
        screen.transform.localScale = new Vector3(0.10f, 0.07f, 0.012f);
        screen.transform.position   = new Vector3(x + 0.06f, 0.99f, 0.445f);
        SetEmissive(screen, new Color(0.3f, 0.9f, 0.4f), 0.5f);
        RemoveCollider(screen);

        _aedStand = aed;
    }

    // ── IV drip stand (background prop) ──────────────────────────────────────
    void BuildIVStand()
    {
        float x = -roomWidth / 2 + 0.6f;
        float z = -roomDepth / 2 + 0.8f;

        var pole = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        pole.name = "IVPole";
        pole.transform.SetParent(transform);
        pole.transform.localScale = new Vector3(0.025f, 0.85f, 0.025f);
        pole.transform.position   = new Vector3(x, 0.85f, z);
        SetMaterial(pole, new Color(0.7f, 0.7f, 0.75f), 0.8f);
        RemoveCollider(pole);

        var hook = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        hook.name = "IVHook";
        hook.transform.SetParent(transform);
        hook.transform.localScale = new Vector3(0.05f, 0.05f, 0.05f);
        hook.transform.position   = new Vector3(x, 1.65f, z);
        SetMaterial(hook, new Color(0.7f, 0.7f, 0.75f), 0.8f);
        RemoveCollider(hook);

        // IV bag
        var bag = GameObject.CreatePrimitive(PrimitiveType.Cube);
        bag.name = "IVBag";
        bag.transform.SetParent(transform);
        bag.transform.localScale = new Vector3(0.1f, 0.18f, 0.03f);
        bag.transform.position   = new Vector3(x, 1.55f, z - 0.02f);
        SetMaterial(bag, new Color(0.75f, 0.90f, 0.95f, 0.6f), 0.6f);
        RemoveCollider(bag);
    }

    // ── Lighting ──────────────────────────────────────────────────────────────
    void BuildLights()
    {
        // Surgical spot above patient — warm focused beam
        var spotGo = new GameObject("SurgicalSpot");
        spotGo.transform.SetParent(transform);
        spotGo.transform.position      = new Vector3(0, roomHeight - 0.1f, 0);
        spotGo.transform.localRotation = Quaternion.Euler(90, 0, 0);
        _surgSpot = spotGo.AddComponent<Light>();
        _surgSpot.type      = LightType.Spot;
        _surgSpot.spotAngle = 55f;
        _surgSpot.range     = roomHeight + 0.5f;
        _surgSpot.intensity = 1.4f;
        _surgSpot.color     = new Color(1f, 0.97f, 0.90f);
        surgicalSpot        = _surgSpot;

        // Ambient fill — soft cool fill from above
        var fillGo = new GameObject("AmbientFill");
        fillGo.transform.SetParent(transform);
        fillGo.transform.position      = new Vector3(0, roomHeight, 0);
        fillGo.transform.localRotation = Quaternion.Euler(90, 0, 0);
        var fill = fillGo.AddComponent<Light>();
        fill.type      = LightType.Directional;
        fill.intensity = 0.5f;
        fill.color     = new Color(0.75f, 0.82f, 1.0f);
        ambientFill    = fill;
    }

    // ── Phase ambient transitions ─────────────────────────────────────────────
    IEnumerator FadeAmbient(Color target, float duration)
    {
        if (ambientFill == null) yield break;
        Color start = ambientFill.color;
        float t = 0f;
        while (t < 1f)
        {
            t += Time.deltaTime / duration;
            ambientFill.color = Color.Lerp(start, target, Mathf.SmoothStep(0, 1, t));
            yield return null;
        }
    }

    IEnumerator ROSCFlash()
    {
        // Green pulse
        Color orig = _surgSpot != null ? _surgSpot.color : Color.white;
        if (_surgSpot != null) _surgSpot.color = new Color(0.3f, 1f, 0.4f);
        float t = 0f;
        while (t < 1f)
        {
            t += Time.deltaTime / 2f;
            if (_surgSpot != null)
                _surgSpot.intensity = Mathf.Lerp(2.5f, 1.4f, t);
            yield return null;
        }
        if (_surgSpot != null) { _surgSpot.color = orig; _surgSpot.intensity = 1.4f; }
    }

    IEnumerator IncorrectFlash()
    {
        if (_surgSpot == null) yield break;
        Color orig = _surgSpot.color;
        _surgSpot.color = new Color(1f, 0.2f, 0.2f);
        yield return new WaitForSeconds(0.25f);
        _surgSpot.color = orig;
    }

    // ── Material helpers ──────────────────────────────────────────────────────
    static void SetMaterial(GameObject go, Color color, float gloss)
    {
        var r = go.GetComponent<Renderer>();
        if (r == null) return;
        var mat = new Material(Shader.Find("Standard"));
        mat.color = color;
        mat.SetFloat("_Glossiness", gloss);
        if (color.a < 1f)
        {
            mat.SetFloat("_Mode", 3);
            mat.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
            mat.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
            mat.SetInt("_ZWrite", 0);
            mat.EnableKeyword("_ALPHABLEND_ON");
            mat.renderQueue = 3000;
        }
        r.material = mat;
    }

    static void SetEmissive(GameObject go, Color color, float intensity)
    {
        var r = go.GetComponent<Renderer>();
        if (r == null) return;
        var mat = new Material(Shader.Find("Standard"));
        mat.color = color;
        mat.EnableKeyword("_EMISSION");
        mat.SetColor("_EmissionColor", color * intensity);
        r.material = mat;
    }

    static void RemoveCollider(GameObject go)
    {
        var col = go.GetComponent<Collider>();
        if (col) Destroy(col);
    }
}
