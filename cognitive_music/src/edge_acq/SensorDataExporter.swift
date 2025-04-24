import Foundation
import Network

/// Very small helper that streams CSV lines over UDP.
/// Fire‑and‑forget: if packets drop we simply lose that sample. ToDo: implement queueing.
final class SensorDataExporter {

    // MARK: ‑‑ Public state
    private(set) var isRunning = false

    // MARK: ‑‑ Private
    private var connection: NWConnection?
    private var host: NWEndpoint.Host = "10.206.121.144"   // <‑‑ change to laptop IP (search in cmd by running ipconfig)
    private let port: NWEndpoint.Port = 9000

    func start() {
        guard !isRunning else { return }

        connection = NWConnection(host: host, port: port, using: .udp)
        connection?.stateUpdateHandler = { state in
            if case .failed(let err) = state { print("UDP failed:", err) }
        }
        connection?.start(queue: .main)
        isRunning = true
    }

    func stop() {
        guard isRunning else { return }
        connection?.cancel()
        connection = nil
        isRunning = false
    }

    /// Send one CSV record, e.g. `"timestamp,ax,ay,az,gx,gy,gz,bpm,temp\n"`
    func send(line: String) {
        guard isRunning, let conn = connection else { return }
        guard let data = line.data(using: .utf8) else { return }

        conn.send(content: data, completion: .contentProcessed { error in
            if let error = error { print("UDP send error:", error) }
        })
    }
}
