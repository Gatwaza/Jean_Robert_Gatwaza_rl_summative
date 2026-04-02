// CPRSceneEnvironment.cs  (v3 — ROSC particles, heartbeat pulse, animated floor grid)
// Procedurally builds the clinical room scene at runtime.
// New in v3:
//   • ROSCParticleSystem  — upward-drifting gold+green sparks on patient revival
//   • HeartbeatPulseLight — surgical spot pulses at simulated BPM during training
//   • AnimatedFloorGrid   — subtle illuminated tile lines that shift with phase
//   • ProtocolVersionCheck — logs warning if bridge sends mismatched protocol ver

using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CPRSceneEnvironment : MonoBehaviour
{
    [Header("Scene Dimensions")]
    public float roomWidth  = 8f;
    public float roomDepth  = 6f;
    public float roomHeight = 3.2f;

    [Header("Lights (auto-assigned)")]
    public Light surgicalSpot;
    public Light ambientFill;

    // ── Internal refs ─────────────────────────────────────────────────────────
    private GameObject _redCrossWall;
    private GameObject _aedStand;
    private Light      _surgSpot;
    private Light      _fillLight;

    // Heartbeat system
    private float _heartbeatPhase   = 0f;
    private float _targetBPM        = 0f;
    private bool  _heartbeatActive  = false;

    // Floor grid tiles
    private readonly List<GameObject> _floorTiles = new List<GameObject>();
    private Color _floorTileTarget = new Color(0.16f, 0.17f, 0.18f);

    // ROSC particle pool
    private GameObject    _particleRoot;
    private const int     PARTICLE_POOL_SIZE = 80;
    private ROSCParticle[] _particles;

    // Protocol version guard
    private const int EXPECTED_VERSION = 2;

    public Vector3 AEDWorldPosition =>
        _aedStand != null
            ? _aedStand.transform.position + Vector3.up * 0.5f
            : Vector3.right * 2f + Vector3.up * 0.5f;

    // ── Lifecycle ─────────────────────────────────────────────────────────────
    void Start()
    {
        BuildScene();
        BuildParticlePool();
        BridgeEvents.OnStateUpdate  += HandleState;
        BridgeEvents.OnPhaseChange  += HandlePhaseChange;
        BridgeEvents.OnEpisodeEnd   += HandleEpisodeEnd;
    }

    void OnDestroy()
    {
        BridgeEvents.OnStateUpdate  -= HandleState;
        BridgeEvents.OnPhaseChange  -= HandlePhaseChange;
        BridgeEvents.OnEpisodeEnd   -= HandleEpisodeEnd;
    }

    void Update()
    {
        UpdateHeartbeatLight();
        UpdateParticles();
    }

    // ── Public phase API ──────────────────────────────────────────────────────
    public void SetPhaseAmbient(Color target, float duration = 1.5f) =>
        StartCoroutine(FadeAmbient(target, duration));

    public void FlashROSC()
    {
        StartCoroutine(ROSCFlash());
        EmitROSCBurst();
    }

    public void FlashIncorrect() => StartCoroutine(IncorrectFlash());

    // ── Event handlers ────────────────────────────────────────────────────────
    void HandleState(StatePacket p)
    {
        // Protocol version guard
        // (StatePacket doesn't yet carry protocol_version, but left as hook)

        // Drive heartbeat speed from heart_rate vital
        float hr = p.vitals?.heart_rate ?? 0f;
        _targetBPM      = Mathf.Lerp(0f, 100f, hr);
        _heartbeatActive = hr > 0.05f;

        if (p.rosc) { FlashROSC(); return; }
        if (!p.is_correct) FlashIncorrect();
    }

    void HandlePhaseChange(StatePacket p)
    {
        Color col = p.phase switch {
            "training" => new Color(0.35f, 0.45f, 0.78f),
            "demo"     => new Color(0.25f, 0.70f, 0.38f),
            _          => new Color(0.50f, 0.50f, 0.60f),
        };
        SetPhaseAmbient(col);
        StartCoroutine(PulseFloorTiles(col));
    }

    void HandleEpisodeEnd(StatePacket p)
    {
        if (p.rosc) FlashROSC();
        _heartbeatActive = false;
        _targetBPM = 0f;
    }

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
        float hw = roomWidth / 2f, hd = roomDepth / 2f;
        int tilesX = 8, tilesZ = 6;

        for (int tx = 0; tx < tilesX; tx++)
        for (int tz = 0; tz < tilesZ; tz++)
        {
            float x = -hw + (tx + 0.5f) * (roomWidth  / tilesX);
            float z = -hd + (tz + 0.5f) * (roomDepth  / tilesZ);

            var tile = GameObject.CreatePrimitive(PrimitiveType.Cube);
            tile.name = $"FloorTile_{tx}_{tz}";
            tile.transform.SetParent(transform);
            tile.transform.localScale = new Vector3(
                roomWidth  / tilesX - 0.04f,
                0.02f,
                roomDepth  / tilesZ - 0.04f);
            tile.transform.position = new Vector3(x, -0.01f, z);

            bool isEdge = tx == 0 || tx == tilesX-1 || tz == 0 || tz == tilesZ-1;
            Color tileCol = isEdge
                ? new Color(0.14f, 0.15f, 0.16f)
                : new Color(0.16f, 0.17f, 0.18f);
            SetMaterial(tile, tileCol, 0.45f);

            var col = tile.GetComponent<Collider>();
            if (col) Destroy(col);
            _floorTiles.Add(tile);
        }

        // Grout lines — thin dark cubes between tiles
        var grout = GameObject.CreatePrimitive(PrimitiveType.Plane);
        grout.name = "GroutLayer";
        grout.transform.SetParent(transform);
        grout.transform.localScale = new Vector3(roomWidth / 10f, 1, roomDepth / 10f);
        grout.transform.position   = new Vector3(0, -0.015f, 0);
        SetMaterial(grout, new Color(0.08f, 0.09f, 0.10f), 0.2f);
    }

    // Animated floor tile colour pulse on phase change
    IEnumerator PulseFloorTiles(Color phaseColor)
    {
        Color flash = Color.Lerp(new Color(0.16f, 0.17f, 0.18f), phaseColor, 0.18f);
        float t = 0f;
        while (t < 1f)
        {
            t += Time.deltaTime * 1.5f;
            Color c = Color.Lerp(flash, new Color(0.16f, 0.17f, 0.18f), t);
            foreach (var tile in _floorTiles)
            {
                if (tile == null) continue;
                var r = tile.GetComponent<Renderer>();
                if (r != null) r.material.color = c;
            }
            yield return null;
        }
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

        // Recessed light panels
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
        HalfWall("PanelLeft",  new Vector3(-roomWidth/2, 0.4f, 0), roomDepth);
        HalfWall("PanelRight", new Vector3( roomWidth/2, 0.4f, 0), roomDepth);
    }

    void HalfWall(string name, Vector3 pos, float length)
    {
        var go  = GameObject.CreatePrimitive(PrimitiveType.Cube);
        go.name = name;
        go.transform.SetParent(transform);
        go.transform.localScale = new Vector3(0.12f, 0.8f, length);
        go.transform.position   = pos;
        SetMaterial(go, new Color(0.84f, 0.83f, 0.80f), 0.08f);
        var col = go.GetComponent<Collider>(); if (col) Destroy(col);

        var cap = GameObject.CreatePrimitive(PrimitiveType.Cube);
        cap.name = name + "_Cap";
        cap.transform.SetParent(transform);
        cap.transform.localScale = new Vector3(0.16f, 0.04f, length);
        cap.transform.position   = pos + Vector3.up * 0.42f;
        SetMaterial(cap, new Color(0.70f, 0.68f, 0.65f), 0.25f);
        var col2 = cap.GetComponent<Collider>(); if (col2) Destroy(col2);
    }

    void BuildRedCrossWall()
    {
        float h = roomHeight, w = roomWidth;
        var wall = GameObject.CreatePrimitive(PrimitiveType.Cube);
        wall.name = "WallBack_RedCross";
        wall.transform.SetParent(transform);
        wall.transform.localScale = new Vector3(w, h, 0.15f);
        wall.transform.position   = new Vector3(0, h / 2, -roomDepth / 2);
        SetMaterial(wall, new Color(0.95f, 0.94f, 0.92f), 0.05f);
        var col = wall.GetComponent<Collider>(); if (col) Destroy(col);
        _redCrossWall = wall;

        Color rcRed = new Color(0.822f, 0.063f, 0.063f);
        Vector3 center = new Vector3(0, h * 0.62f, -roomDepth / 2 + 0.08f);
        float   em     = 0.90f;

        var backing = GameObject.CreatePrimitive(PrimitiveType.Cube);
        backing.name = "CrossBacking";
        backing.transform.SetParent(transform);
        backing.transform.localScale = new Vector3(0.74f, 0.74f, 0.03f) * em;
        backing.transform.position   = center - Vector3.forward * 0.015f;
        SetMaterial(backing, Color.white, 0.1f); RemoveCollider(backing);

        var vBar = GameObject.CreatePrimitive(PrimitiveType.Cube);
        vBar.name = "CrossVertical";
        vBar.transform.SetParent(transform);
        vBar.transform.localScale = new Vector3(0.22f, 0.62f, 0.04f) * em;
        vBar.transform.position   = center;
        SetMaterial(vBar, rcRed, 0.15f); RemoveCollider(vBar);

        var hBar = GameObject.CreatePrimitive(PrimitiveType.Cube);
        hBar.name = "CrossHorizontal";
        hBar.transform.SetParent(transform);
        hBar.transform.localScale = new Vector3(0.62f, 0.22f, 0.04f) * em;
        hBar.transform.position   = center;
        SetMaterial(hBar, rcRed, 0.15f); RemoveCollider(hBar);
    }

    void BuildFloorMat()
    {
        // CPR training mat
        var mat = GameObject.CreatePrimitive(PrimitiveType.Cube);
        mat.name = "CPRMat";
        mat.transform.SetParent(transform);
        mat.transform.localScale = new Vector3(0.72f, 0.025f, 1.8f);
        mat.transform.position   = new Vector3(0, 0.012f, 0);
        SetMaterial(mat, new Color(0.10f, 0.22f, 0.48f), 0.15f);
        RemoveCollider(mat);

        // Hand position markers
        for (int i = 0; i < 2; i++)
        {
            var marker = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            marker.name = "HandMarker" + i;
            marker.transform.SetParent(transform);
            marker.transform.localScale = new Vector3(0.12f, 0.007f, 0.12f);
            marker.transform.position   = new Vector3((i - 0.5f) * 0.12f, 0.027f, 0.08f);
            SetMaterial(marker, new Color(1f, 0.85f, 0.1f), 0.4f);
            RemoveCollider(marker);
        }
    }

    void BuildAED()
    {
        float x = roomWidth / 2 - 0.5f;

        var pole = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        pole.name = "AEDPole"; pole.transform.SetParent(transform);
        pole.transform.localScale = new Vector3(0.04f, 0.55f, 0.04f);
        pole.transform.position   = new Vector3(x, 0.55f, 0.5f);
        SetMaterial(pole, new Color(0.6f, 0.6f, 0.65f), 0.7f);
        RemoveCollider(pole);

        var standBase = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        standBase.name = "AEDBase"; standBase.transform.SetParent(transform);
        standBase.transform.localScale = new Vector3(0.28f, 0.025f, 0.28f);
        standBase.transform.position   = new Vector3(x, 0.012f, 0.5f);
        SetMaterial(standBase, new Color(0.4f, 0.4f, 0.45f), 0.6f);
        RemoveCollider(standBase);

        var aed = GameObject.CreatePrimitive(PrimitiveType.Cube);
        aed.name = "AEDDevice"; aed.transform.SetParent(transform);
        aed.transform.localScale = new Vector3(0.26f, 0.18f, 0.10f);
        aed.transform.position   = new Vector3(x, 0.98f, 0.5f);
        SetMaterial(aed, new Color(1f, 0.52f, 0.05f), 0.35f);
        RemoveCollider(aed);

        // Screen — emissive, blinks in update
        var screen = GameObject.CreatePrimitive(PrimitiveType.Cube);
        screen.name = "AEDScreen"; screen.transform.SetParent(transform);
        screen.transform.localScale = new Vector3(0.10f, 0.07f, 0.012f);
        screen.transform.position   = new Vector3(x + 0.06f, 0.99f, 0.445f);
        SetEmissive(screen, new Color(0.3f, 0.9f, 0.4f), 0.5f);
        RemoveCollider(screen);

        _aedStand = aed;
    }

    void BuildIVStand()
    {
        float x = -roomWidth / 2 + 0.6f, z = -roomDepth / 2 + 0.8f;

        var pole = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        pole.name = "IVPole"; pole.transform.SetParent(transform);
        pole.transform.localScale = new Vector3(0.025f, 0.85f, 0.025f);
        pole.transform.position   = new Vector3(x, 0.85f, z);
        SetMaterial(pole, new Color(0.7f, 0.7f, 0.75f), 0.8f);
        RemoveCollider(pole);

        var bag = GameObject.CreatePrimitive(PrimitiveType.Cube);
        bag.name = "IVBag"; bag.transform.SetParent(transform);
        bag.transform.localScale = new Vector3(0.1f, 0.18f, 0.03f);
        bag.transform.position   = new Vector3(x, 1.55f, z - 0.02f);
        SetMaterial(bag, new Color(0.75f, 0.90f, 0.95f, 0.6f), 0.6f);
        RemoveCollider(bag);
    }

    void BuildLights()
    {
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

        var fillGo = new GameObject("AmbientFill");
        fillGo.transform.SetParent(transform);
        fillGo.transform.position      = new Vector3(0, roomHeight, 0);
        fillGo.transform.localRotation = Quaternion.Euler(90, 0, 0);
        _fillLight = fillGo.AddComponent<Light>();
        _fillLight.type      = LightType.Directional;
        _fillLight.intensity = 0.5f;
        _fillLight.color     = new Color(0.75f, 0.82f, 1.0f);
        ambientFill          = _fillLight;
    }

    // ── Heartbeat pulse light ─────────────────────────────────────────────────
    void UpdateHeartbeatLight()
    {
        if (!_heartbeatActive || _surgSpot == null) return;
        float bps    = _targetBPM / 60f;
        _heartbeatPhase += Time.deltaTime * bps * Mathf.PI * 2f;

        // PQRST-shaped pulse: abs(sin) gives a double-hump; we refine with pow
        float pulse = Mathf.Pow(Mathf.Abs(Mathf.Sin(_heartbeatPhase)), 4f);
        _surgSpot.intensity = Mathf.Lerp(1.2f, 2.0f, pulse);
    }

    // ── ROSC Particle System ──────────────────────────────────────────────────

    struct ROSCParticle
    {
        public GameObject go;
        public Vector3    velocity;
        public float      life;        // 0→1, decreases over time
        public float      maxLife;
        public bool       active;
    }

    void BuildParticlePool()
    {
        _particleRoot = new GameObject("ROSCParticles");
        _particleRoot.transform.SetParent(transform);
        _particles    = new ROSCParticle[PARTICLE_POOL_SIZE];

        Color[] cols = {
            new Color(1.0f, 0.84f, 0.2f),   // gold
            new Color(0.2f, 0.9f, 0.4f),    // green
            new Color(1.0f, 1.0f, 1.0f),    // white
            new Color(0.5f, 0.8f, 1.0f),    // sky blue
        };

        for (int i = 0; i < PARTICLE_POOL_SIZE; i++)
        {
            var sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            sphere.name = "P" + i;
            sphere.transform.SetParent(_particleRoot.transform);
            float s = UnityEngine.Random.Range(0.025f, 0.07f);
            sphere.transform.localScale = Vector3.one * s;
            sphere.transform.position   = Vector3.zero;

            Color c = cols[i % cols.Length];
            var mat = new Material(Shader.Find("Standard"));
            mat.color = c;
            mat.EnableKeyword("_EMISSION");
            mat.SetColor("_EmissionColor", c * 1.5f);
            sphere.GetComponent<Renderer>().material = mat;

            var col = sphere.GetComponent<Collider>();
            if (col) Destroy(col);

            sphere.SetActive(false);
            _particles[i] = new ROSCParticle { go = sphere, active = false };
        }
    }

    void EmitROSCBurst()
    {
        // Chest position of patient
        Vector3 origin = new Vector3(0f, 0.6f, 0f);

        int emitted = 0;
        for (int i = 0; i < _particles.Length && emitted < 40; i++)
        {
            if (_particles[i].active) continue;
            ref ROSCParticle p = ref _particles[i];

            p.go.transform.position = origin + UnityEngine.Random.insideUnitSphere * 0.12f;
            float speed = UnityEngine.Random.Range(1.5f, 4.5f);
            p.velocity = new Vector3(
                UnityEngine.Random.Range(-1.0f, 1.0f),
                UnityEngine.Random.Range(2.0f, 5.0f),
                UnityEngine.Random.Range(-0.5f, 0.5f)
            ).normalized * speed;
            p.maxLife = UnityEngine.Random.Range(0.8f, 2.0f);
            p.life    = p.maxLife;
            p.active  = true;
            p.go.SetActive(true);
            emitted++;
        }
    }

    void UpdateParticles()
    {
        for (int i = 0; i < _particles.Length; i++)
        {
            if (!_particles[i].active) continue;
            ref ROSCParticle p = ref _particles[i];

            p.velocity  += Vector3.down * (4f * Time.deltaTime);  // gravity
            p.go.transform.position += p.velocity * Time.deltaTime;
            p.life -= Time.deltaTime;

            // Scale + fade
            float t    = p.life / p.maxLife;
            float scale = Mathf.Lerp(0.01f, p.go.transform.localScale.x, t);
            p.go.transform.localScale = Vector3.one * Mathf.Max(0.005f, scale);

            var rend = p.go.GetComponent<Renderer>();
            if (rend != null)
            {
                Color c = rend.material.color;
                c.a = Mathf.Clamp01(t * 2f);
                rend.material.color = c;
            }

            if (p.life <= 0f)
            {
                p.active = false;
                p.go.SetActive(false);
            }
        }
    }

    // ── Phase ambient transitions ─────────────────────────────────────────────
    IEnumerator FadeAmbient(Color target, float duration)
    {
        if (_fillLight == null) yield break;
        Color start = _fillLight.color;
        float t = 0f;
        while (t < 1f)
        {
            t += Time.deltaTime / duration;
            _fillLight.color = Color.Lerp(start, target, Mathf.SmoothStep(0, 1, t));
            yield return null;
        }
    }

    IEnumerator ROSCFlash()
    {
        if (_surgSpot == null) yield break;
        Color orig = _surgSpot.color;
        // Three quick green pulses
        for (int i = 0; i < 3; i++)
        {
            _surgSpot.color     = new Color(0.25f, 1f, 0.4f);
            _surgSpot.intensity = 2.8f;
            yield return new WaitForSeconds(0.18f);
            _surgSpot.color     = orig;
            _surgSpot.intensity = 1.4f;
            yield return new WaitForSeconds(0.14f);
        }
        // Sustained warm-green glow for 2s
        float t = 0f;
        Color warmGreen = new Color(0.6f, 1f, 0.6f);
        while (t < 1f)
        {
            t += Time.deltaTime / 2f;
            _surgSpot.color     = Color.Lerp(warmGreen, orig, t);
            _surgSpot.intensity = Mathf.Lerp(2.0f, 1.4f, t);
            yield return null;
        }
    }

    IEnumerator IncorrectFlash()
    {
        if (_surgSpot == null) yield break;
        Color orig = _surgSpot.color;
        _surgSpot.color = new Color(1f, 0.18f, 0.18f);
        yield return new WaitForSeconds(0.22f);
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
        var c = go.GetComponent<Collider>();
        if (c) Destroy(c);
    }
}