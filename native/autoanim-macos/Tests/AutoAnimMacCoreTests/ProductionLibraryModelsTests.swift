import AutoAnimMacCore
import XCTest

final class ProductionLibraryModelsTests: XCTestCase {
    func testDecodesProductionAPIRecordsAndBuildsLiveRelationships() throws {
        let projects = try JSONDecoder().decode(
            ProductionProjectsResponse.self,
            from: Data(
                """
                {"projects":[{
                  "schema_version":"autoanim.production-workspace.v1",
                  "record_type":"project",
                  "project_id":"01abcdefghijklmnopqrstuvwxyz",
                  "name":"Launch Film",
                  "description":"Hero performance",
                  "lifecycle":"active",
                  "created_at":"2026-07-29T10:00:00Z",
                  "updated_at":"2026-07-29T10:00:00Z",
                  "shot_count":1,
                  "shot_ids":["01bcdefghjkmnpqrstvwxyz234"]
                }]}
                """.utf8
            )
        )
        let characters = try JSONDecoder().decode(
            ProductionCharactersResponse.self,
            from: Data(
                """
                {"characters":[{
                  "character_id":"01cdefghjkmnpqrstvwxyz2345",
                  "name":"Mara",
                  "created_at":"2026-07-29T09:00:00Z",
                  "updated_at":"2026-07-29T10:00:00Z",
                  "current_revision_id":"01defghjkmnpqrstvwxyz23456",
                  "consent_status":"active",
                  "consent_scope":"production",
                  "appearance_status":"rgb_atlas_unvalidated",
                  "body_status":"not_attached",
                  "production_validated":false,
                  "material_rights_status":"active",
                  "current_material_rights_expires_at":null
                }]}
                """.utf8
            )
        )
        let shots = try JSONDecoder().decode(
            ProductionShotsResponse.self,
            from: Data(
                """
                {"shots":[{
                  "schema_version":"autoanim.production-workspace.v1",
                  "record_type":"shot",
                  "project_id":"01abcdefghijklmnopqrstuvwxyz",
                  "shot_id":"01bcdefghjkmnpqrstvwxyz234",
                  "name":"Opening",
                  "description":null,
                  "lifecycle":"performance",
                  "character_revision":{
                    "character_id":"01cdefghjkmnpqrstvwxyz2345",
                    "revision_id":"01defghjkmnpqrstvwxyz23456",
                    "revision_manifest_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    "identity_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                    "texture_sha256":null,
                    "texture_uvs_array_sha256":null,
                    "material_manifest_sha256":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
                    "usage_scope":"production"
                  },
                  "created_at":"2026-07-29T10:00:00Z",
                  "updated_at":"2026-07-29T10:00:00Z",
                  "take_count":2,
                  "take_ids":["01efghjkmnpqrstvwxyz234567","01fghjkmnpqrstvwxyz2345678"],
                  "version_count":1,
                  "version_ids":["01ghjkmnpqrstvwxyz23456789"],
                  "latest_version_id":"01ghjkmnpqrstvwxyz23456789"
                }]}
                """.utf8
            )
        )

        let library = ProductionLibrarySnapshot(
            projects: projects.projects,
            characters: characters.characters,
            shots: shots.shots
        )

        XCTAssertEqual(library.project(id: projects.projects[0].id)?.shotCount, 1)
        XCTAssertEqual(library.shots(projectID: projects.projects[0].id).count, 1)
        XCTAssertEqual(library.shots(characterID: characters.characters[0].id).count, 1)
        XCTAssertEqual(library.shots[0].status, .active)
        XCTAssertEqual(library.characters[0].status, .review)
    }

    func testEmptySnapshotRepresentsAnExplicitEmptyAPIState() {
        XCTAssertTrue(ProductionLibrarySnapshot.empty.projects.isEmpty)
        XCTAssertTrue(ProductionLibrarySnapshot.empty.characters.isEmpty)
        XCTAssertTrue(ProductionLibrarySnapshot.empty.shots.isEmpty)
    }
}
