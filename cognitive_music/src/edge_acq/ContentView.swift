struct ContentView: View {
    @EnvironmentObject var data: SensorDataManager

    var body: some View {
        VStack(spacing: 6) {

            // Live vitals list (wrapped in a ScrollView)
            ScrollView {
                sensorList
            }

            // EXPORT CONTROL
            Button(action: { data.toggleExport() }) {
                Label(data.isExporting ? "Stop Export" : "Start Export",
                      systemImage: data.isExporting ? "pause.circle" : "play.circle")
                    .font(.headline)
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .tint(data.isExporting ? .red : .green)

            // Tiny status line
            if data.isExporting {
                Text("Streaming to laptop…")
                    .font(.footnote)
                    .foregroundColor(.yellow)
            }
        }
        .padding()
    }

    @ViewBuilder
    private var sensorList: some View {
        Group {
            // Accelerometer section
            if let a = data.accel?.acceleration {
                Text("Accel  x:\(format(a.x)) y:\(format(a.y)) z:\(format(a.z))")
            }
            // Gyro section
            if let g = data.gyro?.rotationRate {
                Text("Gyro   x:\(format(g.x)) y:\(format(g.y)) z:\(format(g.z))")
            }
            // Heart + Temp
            Text("HR \(format(data.heartRateBPM)) BPM")
            if let t = data.wristTempC {
                Text("Temp \(format(t)) °C")
            }
        }
        .font(.system(size: 14, design: .monospaced))
    }

    private func format(_ v: Double) -> String {
        v.isNaN ? "—" : String(format: "%.2f", v)
    }
}
