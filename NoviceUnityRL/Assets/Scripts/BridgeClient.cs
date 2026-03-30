// BridgeClient.cs
// ================
// Connects to the Python WebSocket bridge and dispatches
// received state packets to all registered listeners.
//
// Setup:
//   1. Install NativeWebSocket via Package Manager:
//      Add to Packages/manifest.json:
//      "com.endel.nativewebsocket": "https://github.com/endel/NativeWebSocket.git#upm"
//   2. Attach this script to an empty GameObject named "BridgeClient"
//   3. Set pythonBridgeUrl in the Inspector (default: ws://localhost:8765)

using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using NativeWebSocket;

// ── JSON data structures ─────────────────────────────────────────────────────

[Serializable]
public class VitalsData
{
    public float heart_rate;
    public float chest_rise;
    public bool  airway_open;
    public int   compressions;
    public float hand_placement;
    public float consciousness;
    public bool  recovery_position;
}

[Serializable]
public class StatePacket
{
    public string type;              // "state" | "phase_change" | "episode_end" | "ping"
    public string phase;             // "random" | "training" | "demo"
    public string algorithm;         // "DQN" | "REINFORCE" | "PPO" | "RANDOM"
    public int    episode;
    public int    step;
    public int    action;
    public string action_name;
    public float  reward;
    public float  cumulative_reward;
    public bool   is_correct;
    public string feedback;
    public int    protocol_stage;
    public float[] landmarks;        // 51 floats: 17 × (x, y, visibility)
    public VitalsData vitals;
    public bool   rosc;
    public float  mean_reward;
    public int    experiment;

    // phase_change fields
    public string message;

    // episode_end fields
    public float total_reward;
    public int   steps;
    public string outcome;
}

// ── Event system ─────────────────────────────────────────────────────────────

public static class BridgeEvents
{
    public static event Action<StatePacket> OnStateUpdate;
    public static event Action<StatePacket> OnPhaseChange;
    public static event Action<StatePacket> OnEpisodeEnd;
    public static event Action<bool>        OnConnectionChanged;

    public static void FireStateUpdate(StatePacket p)    => OnStateUpdate?.Invoke(p);
    public static void FirePhaseChange(StatePacket p)    => OnPhaseChange?.Invoke(p);
    public static void FireEpisodeEnd(StatePacket p)     => OnEpisodeEnd?.Invoke(p);
    public static void FireConnectionChanged(bool conn)  => OnConnectionChanged?.Invoke(conn);
}

// ── Main Bridge Client ────────────────────────────────────────────────────────

public class BridgeClient : MonoBehaviour
{
    [Header("Connection")]
    public string pythonBridgeUrl = "ws://localhost:8765";
    public float  reconnectDelay  = 3f;

    [Header("Debug")]
    public bool   showDebugLog    = true;

    private WebSocket _ws;
    private bool      _isConnected = false;
    private readonly Queue<string> _messageQueue = new Queue<string>();

    // ── Unity lifecycle ───────────────────────────────────────────────────────

    async void Start()
    {
        await ConnectAsync();
    }

    void Update()
    {
        // Dispatch WebSocket callbacks on main thread
#if !UNITY_WEBGL || UNITY_EDITOR
        _ws?.DispatchMessageQueue();
#endif
        // Process received messages on main thread
        while (_messageQueue.Count > 0)
        {
            string raw = _messageQueue.Dequeue();
            ProcessMessage(raw);
        }
    }

    async void OnDestroy()
    {
        if (_ws != null)
            await _ws.Close();
    }

    // ── Connection management ─────────────────────────────────────────────────

    private async System.Threading.Tasks.Task ConnectAsync()
    {
        while (true)
        {
            if (showDebugLog)
                Debug.Log($"[BridgeClient] Connecting to Python bridge: {pythonBridgeUrl}");

            _ws = new WebSocket(pythonBridgeUrl);

            _ws.OnOpen += () => {
                _isConnected = true;
                BridgeEvents.FireConnectionChanged(true);
                Debug.Log("[BridgeClient] ✓ Connected to Python RL bridge");
            };

            _ws.OnMessage += (bytes) => {
                string msg = System.Text.Encoding.UTF8.GetString(bytes);
                lock (_messageQueue) { _messageQueue.Enqueue(msg); }
            };

            _ws.OnError += (err) => {
                Debug.LogWarning($"[BridgeClient] WebSocket error: {err}");
            };

            _ws.OnClose += (code) => {
                _isConnected = false;
                BridgeEvents.FireConnectionChanged(false);
                Debug.Log($"[BridgeClient] Disconnected (code {code}). Retrying in {reconnectDelay}s...");
            };

            await _ws.Connect();

            // Wait for reconnect
            await System.Threading.Tasks.Task.Delay((int)(reconnectDelay * 1000));
            if (!_isConnected)
                continue;

            // Stay connected until closed
            while (_isConnected) {
                await System.Threading.Tasks.Task.Delay(100);
            }
        }
    }

    // ── Message processing ────────────────────────────────────────────────────

    private void ProcessMessage(string raw)
    {
        try
        {
            StatePacket packet = JsonUtility.FromJson<StatePacket>(raw);
            if (packet == null) return;

            if (showDebugLog && packet.type != "ping")
                Debug.Log($"[Bridge] {packet.type} | {packet.phase} | " +
                          $"Action: {packet.action_name} | Reward: {packet.reward:+0.00}");

            switch (packet.type)
            {
                case "state":        BridgeEvents.FireStateUpdate(packet);  break;
                case "phase_change": BridgeEvents.FirePhaseChange(packet);  break;
                case "episode_end":  BridgeEvents.FireEpisodeEnd(packet);   break;
                case "ping":         /* heartbeat — ignore */               break;
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"[BridgeClient] Failed to parse message: {e.Message}\nRaw: {raw}");
        }
    }

    public bool IsConnected => _isConnected;
}
