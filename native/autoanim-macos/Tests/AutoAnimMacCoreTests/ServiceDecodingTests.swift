import Foundation
import Testing
@testable import AutoAnimMacCore

@Suite("Source service contracts")
struct ServiceDecodingTests {
    @Test("Decodes the authenticated ready handshake")
    func decodesReadyEvent() throws {
        let event = try ServiceDecoding.readyEvent(
            from: #"{"event":"ready","url":"http://127.0.0.1:54321/","source_runtime_dependent":true}"#
        )
        #expect(event.event == "ready")
        #expect(event.sourceRuntimeDependent)
        #expect(event.url.port == 54_321)
    }

    @Test("Decodes health without depending on dictionary order")
    func decodesHealth() throws {
        let data = Data(#"{"status":"degraded","checks":{"gnm":{"ready":true,"detail":"3.0"},"a2f_v3_worker":{"ready":false,"detail":"external worker required","version":"3.0"}},"versions":{"autoanim":"0.1.0","python":"3.12.11"}}"#.utf8)
        let health = try ServiceDecoding.health(from: data)
        #expect(health.status == "degraded")
        #expect(health.checks["gnm"]?.ready == true)
        #expect(health.checks["a2f_v3_worker"]?.version == "3.0")
    }

    @Test("Decodes recent jobs and preserves viewer eligibility")
    func decodesJobs() throws {
        let data = Data(#"{"jobs":[{"job_id":"01kxwwdq8gqrsrzycjc3c3kjy9","kind":"audio_animation","status":"succeeded","created_at":"2026-07-19T10:00:00Z","updated_at":"2026-07-19T10:00:01Z","input":{"name":"speech.wav","media_type":"audio/wav"},"warning_count":2,"viewable":true}]}"#.utf8)
        let response = try ServiceDecoding.jobs(from: data)
        #expect(response.jobs.count == 1)
        #expect(response.jobs[0].input.name == "speech.wav")
        #expect(response.jobs[0].viewable)
        #expect(response.jobs[0].warningCount == 2)
    }

    @Test("Requires a full 256-bit lowercase hexadecimal session token")
    func validatesTokenShape() throws {
        let token = try SessionToken.generate()
        #expect(token.count == 64)
        #expect(SessionToken.isValid(token))
        #expect(!SessionToken.isValid(String(repeating: "a", count: 32)))
        #expect(!SessionToken.isValid(String(repeating: "G", count: 64)))
        #expect(throws: SessionTokenError.insufficientEntropy) {
            _ = try SessionToken.generate(byteCount: 15)
        }
    }
}
