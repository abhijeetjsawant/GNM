import Foundation
import Testing
@testable import AutoAnimMacCore

enum ReviewFixture {
    static let jobA = "01kxwwdq8gqrsrzycjc3c3kjy9"
    static let jobB = "01kxy4cqxkzrehcpbfk6jjaar6"
    static let sourcePTS: [Int64] = [0, 1_001, 2_002]

    static func digest(_ label: String) -> String {
        ReviewDigest.sha256(Data(label.utf8))
    }

    static func bundle(
        jobID: String = jobA,
        comparisonVariant: String = "shared"
    ) throws -> ReviewBundle {
        let inputHash = digest("input-\(comparisonVariant)")
        let viewerHash = digest("viewer-\(comparisonVariant)")
        let identityHash = digest("identity-\(comparisonVariant)")
        let timeBase = ReviewRational(1, 30_000)
        func clock(_ hash: String) -> ReviewClock {
            ReviewClock(
                schemaVersion: ReviewContract.clockSchema,
                captureSchemaVersion: "autoanim.capture.v1",
                cursorUnit: "source_pts",
                timeBase: timeBase,
                displayTimeMapping: "(source_pts-first_source_pts)*time_base",
                sourcePTS: sourcePTS,
                frameCount: sourcePTS.count,
                firstSourcePTS: sourcePTS[0],
                lastSourcePTS: sourcePTS[2],
                firstDisplayTimeExactRational: ReviewRational(0, 1),
                sourceStartTimeExactRational: ReviewRational(0, 1),
                durationExactRational: ReviewRational(1_001, 15_000),
                clockSHA256: hash
            )
        }
        let unhashedClock = clock(String(repeating: "0", count: 64))
        let clockHash = ReviewDigest.sha256(
            try ReviewDigest.canonicalJSON(unhashedClock.hashPayload())
        )
        let exactClock = clock(clockHash)
        let artifacts = [
            ReviewArtifact(logicalName: "capture", name: "capture.npz", bytes: 100, sha256: digest("capture"), mediaType: "application/octet-stream", bytesVerified: true),
            ReviewArtifact(logicalName: "controls", name: "performance.npz", bytes: 200, sha256: digest("controls"), mediaType: "application/octet-stream", bytesVerified: true),
            ReviewArtifact(logicalName: "glb", name: "performance.glb", bytes: 300, sha256: digest("glb"), mediaType: "model/gltf-binary", bytesVerified: true),
            ReviewArtifact(logicalName: "viewer_media", name: "source-proxy.mp4", bytes: 400, sha256: viewerHash, mediaType: "video/mp4", bytesVerified: true),
        ]
        func comparison(_ hash: String) throws -> ReviewComparisonKey {
            ReviewComparisonKey(
                schemaVersion: ReviewContract.comparisonSchema,
                inputSHA256: inputHash,
                clockSHA256: clockHash,
                sourcePTSSHA256: try ReviewDigest.sourcePTS(sourcePTS),
                viewerMediaSHA256: viewerHash,
                gnmVersion: "3.0",
                controlsPerformanceSchemaVersion: "autoanim.gnm-performance.v3",
                controlsIdentitySHA256: identityHash,
                comparisonKeySHA256: hash
            )
        }
        let unhashedComparison = try comparison(String(repeating: "0", count: 64))
        let comparisonHash = ReviewDigest.sha256(
            try ReviewDigest.canonicalJSON(unhashedComparison.hashPayload())
        )
        let exactComparison = try comparison(comparisonHash)

        let grouped: [ReviewLayerID: [String]] = [
            .source: ["capture", "viewer_media"],
            .visualBase: ["controls"],
            .audioRepair: [],
            .acting: [],
            .authoredCorrection: [],
            .physics: [],
            .final: ["glb"],
        ]
        var layers = [ReviewLayer]()
        var previous: String?
        for id in ReviewLayerID.allCases {
            let names = grouped[id] ?? []
            let available = !names.isEmpty
            let authority: String
            switch (id, available) {
            case (.source, _): authority = "reference_only"
            case (_, false): authority = "none"
            case (.visualBase, true): authority = "candidate_visual_retarget"
            case (.audioRepair, true): authority = "candidate_lower_face_and_tongue_repair"
            case (.acting, true): authority = "candidate_acting_override"
            case (.authoredCorrection, true): authority = "candidate_bounded_artist_override"
            case (.physics, true): authority = "candidate_simulation"
            case (.final, true): authority = "candidate_composite"
            }
            let consumed = available && [.visualBase, .audioRepair, .acting, .authoredCorrection, .physics].contains(id)
            func layer(_ revision: String) -> ReviewLayer {
                ReviewLayer(
                    schemaVersion: ReviewContract.layerSchema,
                    layerID: id,
                    layerVersion: 1,
                    revisionID: revision,
                    parentRevisionIDs: previous.map { [$0] } ?? [],
                    availability: available ? "available" : "unavailable",
                    artifactLogicalNames: names,
                    motionAuthority: authority,
                    productionMotionAuthority: "none",
                    consumption: ReviewLayerConsumption(
                        consumedByFinalReported: consumed,
                        consumptionIndependentlyVerified: false
                    ),
                    changesMotionReported: available && id != .source,
                    productionValidated: false,
                    approvalStatus: "unapproved"
                )
            }
            let unhashed = layer("review-revision:" + String(repeating: "0", count: 64))
            let revision = "review-revision:" + ReviewDigest.sha256(
                try ReviewDigest.canonicalJSON(unhashed.revisionPayload())
            )
            let exact = layer(revision)
            layers.append(exact)
            previous = exact.revisionID
        }
        let nodes = layers.map {
            ReviewRevisionNode(
                revisionID: $0.revisionID,
                layerID: $0.layerID,
                parentRevisionIDs: $0.parentRevisionIDs,
                immutable: true,
                productionValidated: false,
                approvalStatus: "unapproved"
            )
        }
        let edges = layers.indices.dropFirst().map {
            ReviewRevisionEdge(
                fromRevisionID: layers[$0 - 1].revisionID,
                toRevisionID: layers[$0].revisionID,
                relation: "candidate_layer_composition"
            )
        }
        let graph = ReviewRevisionGraph(
            schemaVersion: ReviewContract.revisionGraphSchema,
            nodes: nodes,
            edges: edges,
            abPairs: [],
            abScope: "cross_bundle_same_comparison_key_only",
            renderableRevisions: [
                ReviewRenderableRevision(
                    revisionID: layers.last!.revisionID,
                    artifactLogicalName: "glb",
                    renderRole: "final_textured_animation",
                    productionValidated: false,
                    approvalStatus: "unapproved"
                )
            ],
            immutable: true,
            undoRedoMode: "append_only_revision_selection",
            productionValidated: false
        )
        let closeups = ["mouth", "tongue", "left_eye", "right_eye"].map {
            ReviewCloseup(
                schemaVersion: ReviewContract.closeupSchema,
                regionID: $0,
                regionVersion: 1,
                selectionSpace: "semantic_face_region",
                normalizedBounds: nil,
                selectionStatus: "native_selection_required",
                renderable: false,
                artifactLogicalNames: ["viewer_media", "glb"],
                productionValidated: false,
                approvalStatus: "unapproved"
            )
        }
        let materialDefinitions = [
            ("base_color", "base_color", "srgb"),
            ("normal", "normal", "linear"),
            ("displacement", "displacement", "linear"),
            ("roughness", "roughness", "linear"),
            ("specular", "specular_color", "linear"),
        ]
        let materials = materialDefinitions.map {
            ReviewMaterialChannel(
                schemaVersion: ReviewContract.materialSchema,
                channel: $0.0,
                manifestKey: $0.1,
                colorSpace: $0.2,
                status: "unavailable",
                artifactLogicalName: nil,
                sha256: nil,
                isolatable: false,
                measured: false,
                productionValidated: false,
                approvalStatus: "unapproved"
            )
        }
        let correction = ReviewCorrectionEligibility(
            schemaVersion: ReviewContract.correctionSchema,
            candidateRequestEligible: false,
            candidateLayerID: .authoredCorrection,
            requiredParentRevisionID: layers.last!.revisionID,
            selectionRequiresExactSourcePTS: true,
            immutableRevisionRequired: true,
            undoRedoMode: "append_only_revision_selection",
            protectedAnchorClasses: ReviewContract.protectedAnchorClasses,
            writerImplemented: false,
            productionRevisionEligible: false,
            humanReviewRecorded: false,
            approvalStatus: "unapproved",
            productionValidated: false,
            reasonCodes: ReviewContract.correctionReasonCodes
        )

        func assemble(_ bundleHash: String) -> ReviewBundle {
            ReviewBundle(
                schemaVersion: ReviewContract.bundleSchema,
                sourceManifest: ReviewSourceManifest(
                    schemaVersion: "1.0",
                    jobID: jobID,
                    kind: "video_performance",
                    status: "succeeded",
                    performanceManifestSHA256: digest("manifest-\(jobID)"),
                    manifestSeal: ReviewManifestSeal(
                        schema: "autoanim.hmac-sha256.v1",
                        keyID: "test-key",
                        signature: digest("signature-\(jobID)"),
                        signatureVerified: false
                    ),
                    input: ReviewSourceInput(
                        name: "performance.mp4",
                        sha256: inputHash,
                        bytes: 1_024,
                        mediaType: "video/mp4",
                        bytesVerified: false
                    )
                ),
                clock: exactClock,
                comparisonKey: exactComparison,
                artifacts: artifacts,
                layers: layers,
                revisionGraph: graph,
                closeups: closeups,
                materialChannels: materials,
                correctionEligibility: correction,
                bridge: ReviewBridgeContract(
                    schemaVersion: ReviewContract.bridgeSchema,
                    allowedMessageTypes: ["cursor", "layer", "selection", "revision"],
                    messageVersionRequired: true,
                    arbitraryScriptMessagesAllowed: false,
                    productionCommandsEnabled: false
                ),
                claims: ReviewClaims(
                    artifactLedgerBytesVerified: true,
                    exactRationalPTSClockVerified: true,
                    manifestSignatureVerified: false,
                    motionConsumptionIndependentlyVerified: false,
                    materialsApproved: false,
                    correctionApproved: false,
                    performanceApproved: false,
                    productionValidated: false,
                    publishable: false
                ),
                limitations: ReviewContract.limitations,
                bundleSHA256: bundleHash
            )
        }
        let unhashedBundle = assemble(String(repeating: "0", count: 64))
        var object = try JSONSerialization.jsonObject(
            with: JSONEncoder().encode(unhashedBundle)
        ) as! [String: Any]
        object.removeValue(forKey: "bundle_sha256")
        return assemble(ReviewDigest.sha256(try ReviewDigest.canonicalJSON(object)))
    }

    static func encoded(_ bundle: ReviewBundle) throws -> Data {
        let object = try JSONSerialization.jsonObject(with: JSONEncoder().encode(bundle))
        return try ReviewDigest.canonicalJSON(object)
    }

    static func mutate(
        _ bundle: ReviewBundle,
        _ body: (inout [String: Any]) throws -> Void,
        rehash: Bool = true
    ) throws -> Data {
        var object = try JSONSerialization.jsonObject(with: encoded(bundle)) as! [String: Any]
        try body(&object)
        if rehash {
            var payload = object
            payload.removeValue(forKey: "bundle_sha256")
            object["bundle_sha256"] = ReviewDigest.sha256(
                try ReviewDigest.canonicalJSON(payload)
            )
        }
        return try ReviewDigest.canonicalJSON(object)
    }
}

@Suite("ReviewBundle v1 native contract")
struct ReviewModelsTests {
    @Test("Strictly decodes the exact source, clock, comparison, and layer contract")
    func decodesValidBundle() throws {
        let source = try ReviewFixture.bundle()
        let decoded = try ReviewBundle.decodeStrict(from: ReviewFixture.encoded(source))
        #expect(decoded == source)
        #expect(decoded.layers.map(\.layerID) == ReviewLayerID.allCases)
        #expect(decoded.revisionGraph.renderableRevisions.count == 1)
        #expect(decoded.revisionGraph.abPairs.isEmpty)
        #expect(!decoded.correctionEligibility.candidateRequestEligible)
    }

    @Test("Rejects duplicate keys even when Foundation would collapse them")
    func rejectsDuplicateKeys() throws {
        let source = try ReviewFixture.bundle()
        let canonical = try ReviewFixture.encoded(source)
        let text = try #require(String(data: canonical, encoding: .utf8))
        let duplicate = Data(
            ("{\"bundle_sha256\":\"\(source.bundleSHA256)\"," + text.dropFirst()).utf8
        )
        #expect(throws: ReviewModelError.self) {
            try ReviewBundle.decodeStrict(from: duplicate)
        }
    }

    @Test("Rejects invalid schemas, job IDs, hashes, and unknown root fields")
    func rejectsInvalidIdentityFields() throws {
        let bundle = try ReviewFixture.bundle()
        let badJob = try ReviewFixture.mutate(bundle) { root in
            var source = root["source_manifest"] as! [String: Any]
            source["job_id"] = "not-a-job"
            root["source_manifest"] = source
        }
        #expect(throws: ReviewModelError.self) { try ReviewBundle.decodeStrict(from: badJob) }

        let badHash = try ReviewFixture.mutate(bundle) { root in
            var artifacts = root["artifacts"] as! [[String: Any]]
            artifacts[0]["sha256"] = String(repeating: "A", count: 64)
            root["artifacts"] = artifacts
        }
        #expect(throws: ReviewModelError.self) { try ReviewBundle.decodeStrict(from: badHash) }

        let unknown = try ReviewFixture.mutate(bundle) { $0["future_claim"] = true }
        #expect(throws: ReviewModelError.self) { try ReviewBundle.decodeStrict(from: unknown) }

        let nestedUnknown = try ReviewFixture.mutate(bundle) { root in
            var artifacts = root["artifacts"] as! [[String: Any]]
            artifacts[0]["future_authority"] = true
            root["artifacts"] = artifacts
        }
        #expect(throws: ReviewModelError.self) {
            try ReviewBundle.decodeStrict(from: nestedUnknown)
        }
    }

    @Test("Rejects unreduced clocks, non-monotonic PTS, and more than 1800 frames")
    func rejectsInvalidClocks() throws {
        let bundle = try ReviewFixture.bundle()
        let unreduced = try ReviewFixture.mutate(bundle) { root in
            var clock = root["clock"] as! [String: Any]
            clock["time_base"] = [2, 60_000]
            root["clock"] = clock
        }
        #expect(throws: ReviewModelError.self) { try ReviewBundle.decodeStrict(from: unreduced) }

        let nonmonotonic = try ReviewFixture.mutate(bundle) { root in
            var clock = root["clock"] as! [String: Any]
            clock["source_pts"] = [0, 1_001, 1_001]
            root["clock"] = clock
        }
        #expect(throws: ReviewModelError.self) { try ReviewBundle.decodeStrict(from: nonmonotonic) }

        let oversized = try ReviewFixture.mutate(bundle) { root in
            var clock = root["clock"] as! [String: Any]
            let values = Array(0...1_800)
            clock["source_pts"] = values
            clock["frame_count"] = values.count
            clock["first_source_pts"] = 0
            clock["last_source_pts"] = 1_800
            root["clock"] = clock
        }
        #expect(throws: ReviewModelError.self) { try ReviewBundle.decodeStrict(from: oversized) }
    }

    @Test("Rejects comparison keys not bound to source, clock, proxy, GNM, and identity")
    func rejectsIncompatibleComparisonKey() throws {
        let bundle = try ReviewFixture.bundle()
        let wrongProxy = try ReviewFixture.mutate(bundle) { root in
            var comparison = root["comparison_key"] as! [String: Any]
            comparison["viewer_media_sha256"] = ReviewFixture.digest("other-proxy")
            root["comparison_key"] = comparison
        }
        #expect(throws: ReviewModelError.self) { try ReviewBundle.decodeStrict(from: wrongProxy) }

        let wrongGNM = try ReviewFixture.mutate(bundle) { root in
            var comparison = root["comparison_key"] as! [String: Any]
            comparison["gnm_version"] = "4.0"
            root["comparison_key"] = comparison
        }
        #expect(throws: ReviewModelError.self) { try ReviewBundle.decodeStrict(from: wrongGNM) }

        let wrongIdentityHashShape = try ReviewFixture.mutate(bundle) { root in
            var comparison = root["comparison_key"] as! [String: Any]
            comparison["controls_identity_sha256"] = "abc"
            root["comparison_key"] = comparison
        }
        #expect(throws: ReviewModelError.self) { try ReviewBundle.decodeStrict(from: wrongIdentityHashShape) }
    }

    @Test("Rejects reordered layers, within-job A/B, multiple renderables, and correction enablement")
    func rejectsUnsupportedCapabilities() throws {
        let bundle = try ReviewFixture.bundle()
        let reordered = try ReviewFixture.mutate(bundle) { root in
            var layers = root["layers"] as! [[String: Any]]
            layers.swapAt(0, 1)
            root["layers"] = layers
        }
        #expect(throws: ReviewModelError.self) { try ReviewBundle.decodeStrict(from: reordered) }

        let abPair = try ReviewFixture.mutate(bundle) { root in
            var graph = root["revision_graph"] as! [String: Any]
            graph["ab_pairs"] = [["a_revision_id": "a", "b_revision_id": "b"]]
            root["revision_graph"] = graph
        }
        #expect(throws: ReviewModelError.self) { try ReviewBundle.decodeStrict(from: abPair) }

        let twoRenderable = try ReviewFixture.mutate(bundle) { root in
            var graph = root["revision_graph"] as! [String: Any]
            var renderables = graph["renderable_revisions"] as! [[String: Any]]
            renderables.append(renderables[0])
            graph["renderable_revisions"] = renderables
            root["revision_graph"] = graph
        }
        #expect(throws: ReviewModelError.self) { try ReviewBundle.decodeStrict(from: twoRenderable) }

        let correctionEnabled = try ReviewFixture.mutate(bundle) { root in
            var correction = root["correction_eligibility"] as! [String: Any]
            correction["candidate_request_eligible"] = true
            root["correction_eligibility"] = correction
        }
        #expect(throws: ReviewModelError.self) { try ReviewBundle.decodeStrict(from: correctionEnabled) }
    }

    @Test("Rejects authority escalation and bundle hash tampering")
    func rejectsClaimEscalationAndTampering() throws {
        let bundle = try ReviewFixture.bundle()
        let escalated = try ReviewFixture.mutate(bundle) { root in
            var claims = root["claims"] as! [String: Any]
            claims["production_validated"] = true
            root["claims"] = claims
        }
        #expect(throws: ReviewModelError.self) { try ReviewBundle.decodeStrict(from: escalated) }

        let tampered = try ReviewFixture.mutate(bundle, { root in
            root["schema_version"] = "autoanim.review-bundle/9.0"
        }, rehash: false)
        #expect(throws: ReviewModelError.self) { try ReviewBundle.decodeStrict(from: tampered) }
    }
}
