"""Interactive local GLB viewer page."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from urllib.parse import urlsplit


VIEWER_THREE_VERSION = "0.183.2"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
VIEWER_VENDOR_FILES = frozenset(
    {
        "three.core.js",
        "three.module.js",
        "addons/controls/OrbitControls.js",
        "addons/loaders/GLTFLoader.js",
        "addons/utils/BufferGeometryUtils.js",
        "addons/utils/SkeletonUtils.js",
        "LICENSE",
    }
)
_AUTOANIM_JOB_ID_LENGTH = 26
_AUTOANIM_JOB_ID_ALPHABET = frozenset("0123456789abcdefghjkmnpqrstvwxyz")
_REVIEW_BRIDGE_REVISION_ALPHABET = frozenset(
    "0123456789abcdefghijklmnopqrstuvwxyz._:-"
)
_REVIEW_BRIDGE_SCHEMA_VERSION = "autoanim.wk-review-bridge/1.0"
_REVIEW_BRIDGE_HANDLER = "autoanimReview"


def default_viewer_vendor_root() -> Path:
    """Return the checksum-pinned local Three.js bundle installed by bootstrap."""

    configured = os.environ.get("AUTOANIM_VIEWER_VENDOR_DIR")
    if configured:
        return Path(configured)
    configured_cache = os.environ.get("AUTOANIM_CACHE_DIR")
    cache = Path(configured_cache) if configured_cache else _PROJECT_ROOT / ".cache/autoanim_gnm"
    return cache / "viewer" / f"three-{VIEWER_THREE_VERSION}"


def viewer_vendor_health(root: str | Path) -> dict[str, str | bool]:
    """Describe whether every runtime module required by the viewer is present."""

    base = Path(root)
    missing = sorted(name for name in VIEWER_VENDOR_FILES if not (base / name).is_file())
    return {
        "ready": not missing,
        "detail": str(base) if not missing else f"missing {', '.join(missing)} under {base}",
        "version": VIEWER_THREE_VERSION,
    }


def _script_json(value: object) -> str:
    """Encode inert JSON for an inline module without permitting ``</script>``."""

    return (
        json.dumps(value, sort_keys=True, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def _validate_review_bridge_job_id(value: str | None) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != _AUTOANIM_JOB_ID_LENGTH
        or any(character not in _AUTOANIM_JOB_ID_ALPHABET for character in value)
    ):
        raise ValueError("Review bridge job ID must be one canonical AutoAnim job ID")
    return value


def _validate_review_bridge_comparison_key(value: str | None) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("Review bridge comparison key must be one lowercase SHA-256")
    return value


def _validate_review_bridge_revision_id(value: str | None) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 128
        or not value[0].isalnum()
        or not value[-1].isalnum()
        or not value.isascii()
        or any(character not in _REVIEW_BRIDGE_REVISION_ALPHABET for character in value)
        or ".." in value
    ):
        raise ValueError("Review bridge revision ID must be one bounded canonical ID")
    return value


def _review_bridge_script(
    job_id: str | None,
    comparison_key: str | None,
    revision_id: str | None,
) -> str:
    """Return the optional bounded WK bridge; ordinary viewers stay unchanged."""

    if job_id is None and comparison_key is None and revision_id is None:
        return ""
    validated_job_id = _validate_review_bridge_job_id(job_id)
    validated_comparison_key = _validate_review_bridge_comparison_key(comparison_key)
    validated_revision_id = _validate_review_bridge_revision_id(revision_id)
    if (
        validated_job_id is None
        or validated_comparison_key is None
        or validated_revision_id is None
    ):
        raise ValueError(
            "Review bridge job ID, comparison key, and revision ID are required together"
        )
    encoded_job_id = _script_json(validated_job_id)
    encoded_comparison_key = _script_json(validated_comparison_key)
    encoded_revision_id = _script_json(validated_revision_id)
    encoded_schema = _script_json(_REVIEW_BRIDGE_SCHEMA_VERSION)
    encoded_handler = _script_json(_REVIEW_BRIDGE_HANDLER)
    return f"""
const nativeReviewBridge=(()=>{{
  const schemaVersion={encoded_schema},jobID={encoded_job_id},comparisonKey={encoded_comparison_key},revisionID={encoded_revision_id},handlerName={encoded_handler},maximumBytes=64*1024,maximumSequence=Number.MAX_SAFE_INTEGER;
  const handler=window.webkit?.messageHandlers?.[handlerName];
  const enabled=Boolean(handler&&typeof handler.postMessage==='function');
  const exactKeys=(value,keys)=>{{
    if(!value||typeof value!=='object'||Array.isArray(value))return false;
    const actual=Object.keys(value).sort(),expected=[...keys].sort();
    return actual.length===expected.length&&actual.every((key,index)=>key===expected[index])
  }};
  const boundedJSON=value=>{{
    try{{const encoded=JSON.stringify(value);return typeof encoded==='string'&&new TextEncoder().encode(encoded).byteLength<=maximumBytes?encoded:null}}catch(_error){{return null}}
  }};
  const validSequence=value=>Number.isSafeInteger(value)&&value>=0&&value<=maximumSequence;
  const decimalInteger=/^-?(0|[1-9][0-9]*)$/;
  const validDecimal=value=>typeof value==='string'&&decimalInteger.test(value);
  const validLayerIDs=new Set(['surface','wireframe','tracker','pixelROI','exactSourceFrame']);
  const validRegionSelections=new Set(['none','mouth','eyes','upperFace','head']);
  const validCameraSelections=new Set(['home','front']);
  let outboundSequence=0,lastInboundSequence=-1,pendingCursor=null,rendererReady=false,readySent=false;
  const layerState={{surface:true,wireframe:false,tracker:true,pixelROI:true,exactSourceFrame:true}};
  let selectionState={{kind:'region',value:'none'}};
  function post(type,payload){{
    if(!enabled||outboundSequence>maximumSequence)return false;
    const envelope={{schemaVersion,sequence:outboundSequence,type,jobID,payload}};
    if(boundedJSON(envelope)===null)return false;
    outboundSequence+=1;
    try{{handler.postMessage(envelope);return true}}catch(_error){{return false}}
  }}
  function reportError(code,detail,recoverable=true){{
    const boundedDetail=String(detail??'').slice(0,512);
    post('viewer.error',{{code:String(code).slice(0,64),detail:boundedDetail,recoverable:Boolean(recoverable)}})
  }}
  function verificationState(){{
    if(presentationClockState==='verified_static')return 'server_decoded';
    if(presentationClockState==='verified')return 'presented_frame';
    return 'fallback'
  }}
  function cursorPayload(index,reason){{
    const frame=evidenceFrames[index],tick=frame?.projectTickExactRational;
    if(!frame||!Array.isArray(tick)||tick.length!==2)return null;
    return {{
      frameIndex:index,
      sourcePTS:String(frame.sourcePTS),
      projectTick:[String(tick[0]),String(tick[1])],
      verification:verificationState(),
      reason,
    }}
  }}
  function emitCursor(index,reason='playback'){{
    if(!enabled||!Number.isSafeInteger(index)||index<0||index>=evidenceFrames.length)return false;
    const payload=cursorPayload(index,reason);
    return payload!==null&&post('cursor.changed',payload)
  }}
  function acknowledgeExactCursor(index){{
    const command=pendingCursor,frame=evidenceFrames[index];
    if(
      !command||command.frameIndex!==index||!frame||media?.paused!==true
      ||staticReviewFrameIndex!==index||presentationClockState!=='verified_static'
      ||String(frame.sourcePTS)!==command.expectedSourcePTS
    )return false;
    pendingCursor=null;
    return post('cursor.applied',{{
      requestSequence:command.sequence,
      frameIndex:index,
      sourcePTS:String(frame.sourcePTS),
      verification:'server_decoded',
    }})
  }}
  function applyCursor(payload,sequence){{
    if(
      !exactKeys(payload,['frameIndex','expectedSourcePTS','operation'])
      ||!Number.isSafeInteger(payload.frameIndex)||payload.frameIndex<0
      ||!validDecimal(payload.expectedSourcePTS)
      ||!['seek','step','pause'].includes(payload.operation)
    )throw new Error('cursor.set payload is invalid');
    if(
      !evidenceReady||!diagnosticsReady||!reviewFrameUrlTemplate||!media
      ||payload.frameIndex>=evidenceFrames.length
      ||String(evidenceFrames[payload.frameIndex]?.sourcePTS)!==payload.expectedSourcePTS
    )throw new Error('cursor.set is not bound to an exact review frame');
    media.pause();
    pendingCursor={{sequence,frameIndex:payload.frameIndex,expectedSourcePTS:payload.expectedSourcePTS}};
    pendingEvidenceFrameIndex=payload.frameIndex;
    pendingPresentationAttempt=0;
    presentationClockState='waiting';
    refreshEvidenceControls();
    void displayExactReviewFrame(payload.frameIndex)
  }}
  function applyRenderLayers(){{
    if(root)root.visible=layerState.surface||layerState.wireframe;
    const mode=layerState.surface?(layerState.wireframe?'surface-wire':'surface'):'wire';
    const selector=document.querySelector('#mode');if(selector)selector.value=mode;
    applyMode()
  }}
  function applyLayer(payload){{
    if(
      !exactKeys(payload,['layerID','visible'])
      ||!validLayerIDs.has(payload.layerID)||typeof payload.visible!=='boolean'
    )throw new Error('layer.set payload is invalid');
    layerState[payload.layerID]=payload.visible;
    if(payload.layerID==='surface'||payload.layerID==='wireframe')applyRenderLayers();
    else if(payload.layerID==='tracker')evidencePanel.hidden=!payload.visible;
    else if(payload.layerID==='pixelROI'){{
      if(diagnosticOverlay)diagnosticOverlay.hidden=!payload.visible;
      if(payload.visible&&diagnosticsReady&&displayedEvidenceFrameIndex>=0)drawDiagnosticOverlay(diagnosticFrames[displayedEvidenceFrameIndex])
    }}else if(payload.layerID==='exactSourceFrame'&&reviewFrameImage){{
      reviewFrameImage.hidden=!payload.visible||staticReviewFrameIndex<0
    }}
    post('layer.changed',{{layerID:payload.layerID,visible:payload.visible}})
  }}
  function applySelection(payload){{
    if(!exactKeys(payload,['kind','value'])||typeof payload.kind!=='string'||typeof payload.value!=='string')throw new Error('selection.set payload is invalid');
    if(payload.kind==='region'){{
      if(!validRegionSelections.has(payload.value))throw new Error('selection.set region is invalid')
    }}else if(payload.kind==='cameraPreset'){{
      if(!validCameraSelections.has(payload.value))throw new Error('selection.set camera preset is invalid');
      camera.position.copy(homePosition);controls.target.copy(homeTarget);controls.update()
    }}else throw new Error('selection.set kind is invalid');
    selectionState={{kind:payload.kind,value:payload.value}};
    if(diagnosticsReady&&displayedEvidenceFrameIndex>=0)drawDiagnosticOverlay(diagnosticFrames[displayedEvidenceFrameIndex]);
    post('selection.changed',selectionState)
  }}
  function applyRevision(payload){{
    if(!exactKeys(payload,['comparisonKey','revisionID'])||payload.comparisonKey!==comparisonKey||payload.revisionID!==revisionID)throw new Error('revision.set payload is invalid');
    post('revision.ready',{{comparisonKey,revisionID}})
  }}
  function receive(envelope){{
    if(!enabled)return false;
    if(
      boundedJSON(envelope)===null
      ||!exactKeys(envelope,['schemaVersion','sequence','type','jobID','payload'])
      ||envelope.schemaVersion!==schemaVersion||envelope.jobID!==jobID
      ||!validSequence(envelope.sequence)||typeof envelope.type!=='string'
    ){{reportError('BRIDGE_ENVELOPE_INVALID','Native review bridge envelope was rejected.');return false}}
    if(envelope.sequence<=lastInboundSequence)return false;
    try{{
      if(envelope.type==='cursor.set')applyCursor(envelope.payload,envelope.sequence);
      else if(envelope.type==='layer.set')applyLayer(envelope.payload);
      else if(envelope.type==='selection.set')applySelection(envelope.payload);
      else if(envelope.type==='revision.set')applyRevision(envelope.payload);
      else throw new Error('Native review bridge message type is not allowed');
      lastInboundSequence=envelope.sequence;
      return true
    }}catch(error){{reportError('BRIDGE_PAYLOAD_INVALID',error.message);return false}}
  }}
  function maybeReady(){{
    if(!enabled||readySent||!rendererReady||!evidenceReady)return false;
    readySent=true;
    const exactCursor=Boolean(diagnosticsReady&&reviewFrameUrlTemplate);
    post('viewer.ready',{{comparisonKey,revisionID,frameCount:evidenceFrames.length,capabilities:['layer','selection','revision',...(exactCursor?['cursor']:[])]}});
    post('revision.ready',{{comparisonKey,revisionID}});
    return true
  }}
  function setRendererReady(){{rendererReady=true;maybeReady()}}
  function evidenceDidLoad(){{maybeReady()}}
  function exactFrameDidLoad(index){{
    if(reviewFrameImage)reviewFrameImage.hidden=!layerState.exactSourceFrame;
    acknowledgeExactCursor(index)
  }}
  function modeDidChange(mode){{
    layerState.surface=mode!=='wire';layerState.wireframe=mode!=='surface';
    if(root)root.visible=true;
    post('layer.changed',{{layerID:'surface',visible:layerState.surface}});
    post('layer.changed',{{layerID:'wireframe',visible:layerState.wireframe}})
  }}
  function selectedRegion(){{return selectionState.kind==='region'&&selectionState.value!=='none'?selectionState.value:null}}
  function shutdown(){{pendingCursor=null}}
  const entrypoint=Object.freeze({{receive}});
  Object.defineProperty(window,'autoanimReview',{{value:entrypoint,writable:false,configurable:false,enumerable:false}});
  return Object.freeze({{enabled,emitCursor,reportError,setRendererReady,evidenceDidLoad,exactFrameDidLoad,modeDidChange,selectedRegion,shutdown}})
}})();
"""


def _validate_performance_evidence_url(value: str | None) -> str | None:
    """Accept only the job artifact route used by the server allowlist."""

    if value is None:
        return None
    parsed = urlsplit(value)
    parts = parsed.path.split("/")
    if (
        parsed.scheme
        or parsed.netloc
        or parsed.query
        or parsed.fragment
        or len(parts) != 6
        or parts[1:3] != ["api", "jobs"]
        or not parts[3]
        or not all(
            character.isascii()
            and (character.isalnum() or character in {"-", "_"})
            for character in parts[3]
        )
        or parts[4:] != ["files", "performance-evidence.json"]
    ):
        raise ValueError(
            "Performance evidence must use the allowlisted job artifact URL"
        )
    return value


def _validate_observation_v3_url(value: str | None) -> str | None:
    """Accept only the derived, authenticated Observation-v3 review route."""

    if value is None:
        return None
    parsed = urlsplit(value)
    parts = parsed.path.split("/")
    if (
        parsed.scheme
        or parsed.netloc
        or parsed.query
        or parsed.fragment
        or len(parts) != 5
        or parts[1:3] != ["api", "jobs"]
        or not parts[3]
        or not all(
            character.isascii()
            and (character.isalnum() or character in {"-", "_"})
            for character in parts[3]
        )
        or parts[4] != "observation-v3-view"
    ):
        raise ValueError(
            "Observation-v3 diagnostics must use the allowlisted job review URL"
        )
    return value


def viewer_html(
    *,
    asset_url: str,
    title: str,
    media_url: str | None = None,
    media_type: str | None = None,
    performance_evidence_url: str | None = None,
    observation_v3_url: str | None = None,
    metadata: dict | None = None,
    vendor_base_url: str = f"/api/viewer/vendor/{VIEWER_THREE_VERSION}",
    review_bridge_job_id: str | None = None,
    review_bridge_comparison_key: str | None = None,
    review_bridge_revision_id: str | None = None,
) -> str:
    """Return a self-contained shell for allowlisted GLB, media, and evidence."""

    safe_title = html.escape(title)
    encoded_url = _script_json(asset_url)
    encoded_media_url = _script_json(media_url)
    encoded_media_kind = _script_json(
        "video" if media_type is not None and media_type.startswith("video/") else "audio"
    )
    evidence_url = _validate_performance_evidence_url(performance_evidence_url)
    observation_url = _validate_observation_v3_url(observation_v3_url)
    if observation_url is not None and evidence_url is None:
        raise ValueError("Observation-v3 review requires tracker evidence")
    if observation_url is not None and evidence_url is not None:
        evidence_job = urlsplit(evidence_url).path.split("/")[3]
        observation_job = urlsplit(observation_url).path.split("/")[3]
        if evidence_job != observation_job:
            raise ValueError(
                "Observation-v3 review and tracker evidence must belong to the same job"
            )
    review_layers = []
    if evidence_url is not None:
        review_layers.append(
            {
                "id": "tracker-regions",
                "kind": "regional_tracker_evidence",
                "schemaVersion": "autoanim.performance-evidence.v2",
                "url": evidence_url,
                "clockBinding": "source_media_pts",
                "authority": "observation_only",
            }
        )
    if observation_url is not None:
        job_prefix = observation_url.removesuffix("/observation-v3-view")
        review_layers.append(
            {
                "id": "pixel-regions",
                "kind": "regional_pixel_roi_evidence",
                "schemaVersion": "autoanim.observation-v3-view/1.0",
                "url": observation_url,
                "clockBinding": "source_media_presented_frame",
                "coordinateSpace": "display_oriented_rgb_pixels",
                "authority": "observation_only",
                "requiresDisplayBinding": True,
            }
        )
        review_layers.append(
            {
                "id": "exact-display-proxy-frame",
                "kind": "display_proxy_frame_review_image",
                "schemaVersion": "autoanim.review-frame.png/1.0",
                "urlTemplate": (
                    f"{job_prefix}/review-frames/{{frameIndex}}.png"
                ),
                "clockBinding": "source_media_presented_frame",
                "coordinateSpace": "display_oriented_rgb_pixels",
                "authority": "review_pixels",
                "requiresDisplayBinding": True,
            }
        )
    encoded_review_layers = _script_json(
        {
            "schemaVersion": "autoanim.viewer-review-layers/1.0",
            "layers": review_layers,
        }
    )
    encoded_metadata = _script_json(metadata)
    encoded_three_url = _script_json(f"{vendor_base_url}/three.module.js")
    encoded_addons_url = _script_json(f"{vendor_base_url}/addons/")
    review_bridge_script = _review_bridge_script(
        review_bridge_job_id,
        review_bridge_comparison_key,
        review_bridge_revision_id,
    )
    bridge_enabled = review_bridge_job_id is not None
    bridge_cursor_hook = (
        "nativeReviewBridge.emitCursor(bounded,presentationClockState==='verified_static'?'seek':'playback');"
        if bridge_enabled
        else ""
    )
    bridge_exact_frame_hook = (
        "nativeReviewBridge.exactFrameDidLoad(index);"
        if bridge_enabled
        else ""
    )
    bridge_evidence_hook = (
        "nativeReviewBridge.evidenceDidLoad();"
        if bridge_enabled
        else ""
    )
    mode_change_listener = (
        "()=>{applyMode();nativeReviewBridge.modeDidChange(document.querySelector('#mode').value)}"
        if bridge_enabled
        else "applyMode"
    )
    bridge_renderer_ready_hook = (
        "nativeReviewBridge.setRendererReady();"
        if bridge_enabled
        else ""
    )
    bridge_renderer_error_hook = (
        "nativeReviewBridge.reportError('GLB_LOAD_FAILED',error.message,false);"
        if bridge_enabled
        else ""
    )
    bridge_evidence_error_hook = (
        "nativeReviewBridge.reportError('EVIDENCE_LOAD_FAILED',error.message,true);"
        if bridge_enabled
        else ""
    )
    bridge_diagnostics_error_hook = (
        "nativeReviewBridge.reportError('DIAGNOSTICS_LOAD_FAILED',error.message,true);"
        if bridge_enabled
        else ""
    )
    bridge_exact_frame_error_hook = (
        "nativeReviewBridge.reportError('EXACT_FRAME_LOAD_FAILED',error.message,true);"
        if bridge_enabled
        else ""
    )
    bridge_shutdown_hook = (
        "nativeReviewBridge.shutdown();"
        if bridge_enabled
        else ""
    )
    bridge_selected_region_declaration = (
        "const selectedRegion=nativeReviewBridge.selectedRegion();"
        if bridge_enabled
        else ""
    )
    bridge_region_alpha_hook = (
        "context.globalAlpha=selectedRegion===null||selectedRegion===regionName?1:.25;"
        if bridge_enabled
        else ""
    )
    bridge_region_alpha_reset = (
        "context.globalAlpha=1;" if bridge_enabled else ""
    )
    return f"""<!doctype html>
<html lang="en"><head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{safe_title} · 3D viewer</title>
  <style>
    :root{{color-scheme:dark;--bg:#090b0d;--panel:#13171b;--line:#303840;--text:#f2f4f5;--muted:#9da8b1;--accent:#d8ff63}}
    *{{box-sizing:border-box}}html,body{{height:100%;margin:0;background:var(--bg);color:var(--text);font:14px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace}}
    main{{height:100%;display:grid;grid-template-columns:minmax(220px,290px) 1fr}}
    aside{{padding:24px;background:var(--panel);border-right:1px solid var(--line);overflow:auto}}
    h1{{font:700 28px/1 system-ui,sans-serif;letter-spacing:-.04em;margin:0 0 10px}}p{{color:var(--muted)}}
    label{{display:block;margin:22px 0 7px;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.08em}}
    select,input,button,audio,video{{width:100%;padding:11px;border-radius:9px;border:1px solid var(--line);background:#0b0e11;color:var(--text);font:inherit}}
    .media-shell{{position:relative;width:100%;margin-top:0}}.media-shell video{{display:block;width:100%;max-height:240px;object-fit:contain;padding:0}}
    .media-shell .exact-review-frame{{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;background:#0b0e11;pointer-events:none}}.media-shell canvas{{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}}
    button{{margin-top:12px;background:var(--accent);color:#111500;border:0;font-weight:800;cursor:pointer}}
    button:disabled{{cursor:not-allowed;opacity:.45}}.evidence-review{{margin-top:18px;padding:12px;border:1px solid var(--line);border-radius:10px;background:#0b0e11}}
    .evidence-review h2{{margin:0 0 8px;font:700 13px/1.2 system-ui,sans-serif}}.frame-nav{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}
    .frame-nav button{{width:100%;margin:0;padding:8px}}#evidence-readout{{margin:9px 0 6px;color:var(--text);font-size:11px}}
    #evidence-flag{{padding:7px 8px;border-radius:7px;background:#252c32;color:var(--muted);font-size:11px}}#evidence-flag.missing{{background:#401f25;color:#ffb3bd}}#evidence-flag.unknown{{background:#322f1b;color:#f3dfa1}}
    #evidence-regions{{display:grid;gap:7px;margin:9px 0 0}}#evidence-regions div{{display:grid;grid-template-columns:74px 1fr;gap:10px;font-size:11px}}#evidence-regions dt{{color:var(--muted)}}#evidence-regions dd{{margin:0;text-align:right;white-space:pre-line;overflow-wrap:anywhere}}#evidence-diagnostic-state{{margin:9px 0 0;font-size:10px}}#evidence-error{{margin:8px 0 0;color:#ff9da8;font-size:11px}}
    #stage{{min-width:0;min-height:0;position:relative}}canvas{{display:block;width:100%;height:100%}}
    #status{{position:absolute;left:18px;bottom:16px;padding:8px 11px;background:#080a0cdd;border:1px solid var(--line);border-radius:8px;color:var(--muted)}}
    .legend{{margin-top:24px;padding-top:16px;border-top:1px solid var(--line);font-size:12px}}#metadata{{white-space:pre-wrap;overflow-wrap:anywhere;color:var(--muted)}}
    @media(max-width:720px){{main{{grid-template-columns:1fr;grid-template-rows:auto 1fr}}aside{{padding:14px;border-right:0;border-bottom:1px solid var(--line)}}aside>.viewer-intro,.legend{{display:none}}#evidence-diagnostic-state,#evidence-error{{display:block}}h1{{font-size:20px}}label{{display:inline-block;margin:6px 6px 4px 0}}select,input,button{{width:auto}}}}
  </style>
  <script type="importmap">{{"imports":{{"three":{encoded_three_url},"three/addons/":{encoded_addons_url}}}}}</script>
</head><body><main>
  <aside><h1>{safe_title}</h1><p class="viewer-intro">Orbit, zoom, and inspect the exact seam-correct GNM asset exported by this job.</p>
    <label for="mode">Display</label><select id="mode"><option value="surface">Surface / texture</option><option value="surface-wire">Surface + topology</option><option value="wire">Topology only</option></select>
    <label for="exposure">Exposure</label><input id="exposure" type="range" min="0.35" max="2.25" step="0.05" value="1.0">
    <button id="reset" type="button">Reset camera</button><div id="media-slot"></div>
    <section id="evidence-panel" class="evidence-review" aria-label="Performance evidence review" hidden><h2>Source-frame evidence</h2>
      <div class="frame-nav"><button id="previous-source-frame" type="button" disabled>Previous frame</button><button id="next-source-frame" type="button" disabled>Next frame</button></div>
      <div id="evidence-readout" aria-live="polite">Evidence unavailable</div><div id="evidence-flag" class="unknown">UNKNOWN</div>
      <dl id="evidence-regions"><div><dt>Mouth</dt><dd id="evidence-mouth">—</dd></div><div><dt>Eyes</dt><dd id="evidence-eyes">—</dd></div><div><dt>Upper face</dt><dd id="evidence-upper-face">—</dd></div><div><dt>Head</dt><dd id="evidence-head">—</dd></div></dl>
      <p id="evidence-diagnostic-state">Pixel diagnostics unavailable · tracker evidence only</p>
      <p id="evidence-error" role="alert" hidden></p>
    </section>
    <div class="legend"><strong>GNM Head 3.0</strong><p id="metrics">Loading geometry…</p><p>Texture seams are split into duplicate render vertices that retain a mapping to the original GNM topology.</p><div id="metadata"></div></div>
  </aside>
  <section id="stage" aria-label="Interactive 3D GNM head"><div id="status" role="status" aria-live="polite">Loading 3D asset…</div></section>
</main>
<script type="module">
import * as THREE from 'three';
import {{OrbitControls}} from 'three/addons/controls/OrbitControls.js';
import {{GLTFLoader}} from 'three/addons/loaders/GLTFLoader.js';

const assetUrl={encoded_url},mediaUrl={encoded_media_url},mediaKind={encoded_media_kind},reviewManifest={encoded_review_layers},reviewLayers=reviewManifest.layers,performanceEvidenceUrl=reviewLayers.find(layer=>layer.kind==='regional_tracker_evidence')?.url??null,observationV3Url=reviewLayers.find(layer=>layer.kind==='regional_pixel_roi_evidence')?.url??null,reviewFrameUrlTemplate=reviewLayers.find(layer=>layer.kind==='display_proxy_frame_review_image')?.urlTemplate??null,metadata={encoded_metadata},stage=document.querySelector('#stage'),status=document.querySelector('#status');
if(metadata){{const lines=[];for(const [label,value] of Object.entries(metadata))lines.push(`${{label.replaceAll('_',' ')}}: ${{Array.isArray(value)?value.join(', '):String(value)}}`);document.querySelector('#metadata').textContent=lines.join('\\n')}}
const scene=new THREE.Scene();scene.background=new THREE.Color(0x0b0e10);
const camera=new THREE.PerspectiveCamera(28,1,.005,20);
let renderer;try{{renderer=new THREE.WebGLRenderer({{antialias:true,alpha:false}})}}catch(error){{status.textContent='WebGL is unavailable. Download the GLB from the job result to inspect it in another viewer.';status.style.color='#ff7d7d';throw error}}renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1;renderer.domElement.setAttribute('role','img');renderer.domElement.setAttribute('aria-label','Interactive 3D GNM head');stage.prepend(renderer.domElement);
const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.dampingFactor=.075;controls.screenSpacePanning=true;controls.enablePan=false;controls.minPolarAngle=.2;controls.maxPolarAngle=Math.PI-.2;
scene.add(new THREE.HemisphereLight(0xeaf3ff,0x33291f,2.2));const key=new THREE.DirectionalLight(0xffffff,3.1);key.position.set(.8,.7,1.4);scene.add(key);const rim=new THREE.DirectionalLight(0xb7d8ff,1.8);rim.position.set(-1,.35,-1);scene.add(rim);
const grid=new THREE.GridHelper(1,20,0x4c555d,0x252b30);scene.add(grid);
let root=null,mixer=null,animationAction=null,animationDuration=0,homePosition=new THREE.Vector3(),homeTarget=new THREE.Vector3();
let media=null,diagnosticOverlay=null,reviewFrameImage=null,mediaShell=null;if(mediaUrl){{const label=document.createElement('label');label.textContent=mediaKind==='video'?'Source performance':'Playback';media=document.createElement(mediaKind);media.controls=true;media.preload=mediaKind==='video'?'auto':'metadata';if(mediaKind==='video'){{media.playsInline=true;mediaShell=document.createElement('div');mediaShell.className='media-shell';reviewFrameImage=document.createElement('img');reviewFrameImage.className='exact-review-frame';reviewFrameImage.alt='';reviewFrameImage.hidden=true;diagnosticOverlay=document.createElement('canvas');diagnosticOverlay.setAttribute('aria-hidden','true');mediaShell.append(media,reviewFrameImage,diagnosticOverlay)}}media.src=mediaUrl;document.querySelector('#media-slot').append(label,mediaShell||media)}}
const evidencePanel=document.querySelector('#evidence-panel');
const evidenceReadout=document.querySelector('#evidence-readout');
const evidenceFlag=document.querySelector('#evidence-flag');
const evidenceError=document.querySelector('#evidence-error');
const evidenceDiagnosticState=document.querySelector('#evidence-diagnostic-state');
const previousSourceFrame=document.querySelector('#previous-source-frame');
const nextSourceFrame=document.querySelector('#next-source-frame');
const evidenceRegionElements={{
  mouth:document.querySelector('#evidence-mouth'),
  eyes:document.querySelector('#evidence-eyes'),
  upperFace:document.querySelector('#evidence-upper-face'),
  head:document.querySelector('#evidence-head'),
}};
const requiredEvidenceRegions=['mouth','eyes','upperFace','head'];
let evidenceFrames=[],diagnosticFrames=[],diagnosticSource=null,diagnosticDisplay=null,performanceEvidenceSource=null,displayedEvidenceFrameIndex=-1,displayedDiagnosticsReady=false,evidenceReady=false,diagnosticsReady=false,presentedMediaTime=null,pendingEvidenceFrameIndex=-1,pendingPresentationAttempt=0,pendingPresentationTimer=null,videoFrameCallbackId=null,presentationClockState='waiting',forcePauseAfterPresentation=false,mutedBeforeForcedPresentation=false,staticReviewFrameIndex=-1,staticReviewInternalSeekTime=null,reviewFrameObjectUrl=null,reviewFrameAbortController=null;const evidenceAbortController=new AbortController();
const supportsPresentedFrameClock=Boolean(media&&mediaKind==='video'&&typeof media.requestVideoFrameCallback==='function');
{review_bridge_script}
function validateEvidenceArtifactUrl(value){{
  const parsed=new URL(value,window.location.href);
  const allowlisted=/^\\/api\\/jobs\\/[A-Za-z0-9_-]+\\/files\\/performance-evidence\\.json$/;
  if(parsed.origin!==window.location.origin||!allowlisted.test(parsed.pathname)||parsed.search||parsed.hash){{
    throw new Error('Evidence URL is not an allowlisted same-origin job artifact')
  }}
  return parsed.href
}}
function validateObservationV3Url(value){{
  const parsed=new URL(value,window.location.href);
  const allowlisted=/^\\/api\\/jobs\\/[A-Za-z0-9_-]+\\/observation-v3-view$/;
  if(parsed.origin!==window.location.origin||!allowlisted.test(parsed.pathname)||parsed.search||parsed.hash){{
    throw new Error('Observation-v3 URL is not an allowlisted same-origin job review route')
  }}
  return parsed.href
}}
function reviewFrameUrl(index){{
  if(!reviewFrameUrlTemplate||!Number.isSafeInteger(index)||index<0)throw new Error('Exact review frame URL is unavailable');
  const parsed=new URL(reviewFrameUrlTemplate.replace('{{frameIndex}}',String(index)),window.location.href),allowlisted=/^\\/api\\/jobs\\/[A-Za-z0-9_-]+\\/review-frames\\/[0-9]+\\.png$/;
  if(parsed.origin!==window.location.origin||!allowlisted.test(parsed.pathname)||parsed.search||parsed.hash)throw new Error('Exact review frame URL is not allowlisted');
  return parsed.href
}}
function validatePerformanceEvidence(payload){{
  if(!payload||payload.schemaVersion!=='autoanim.performance-evidence.v2'||payload.sourceMode!=='video_follow'||payload.consumedByRetargeting!==false){{
    throw new Error('Unsupported performance-evidence contract')
  }}
  if(!Array.isArray(payload.frames)||!payload.frames.length||payload.source?.frameCount!==payload.frames.length){{
    throw new Error('Performance evidence has an invalid frame list')
  }}
  let previousTimestamp=-Infinity,previousPTS=null,previousProjectTick=null;
  for(let index=0;index<payload.frames.length;index++){{
    const frame=payload.frames[index],exactTick=frame?.projectTickExactRational;
    const exactTickValid=Array.isArray(exactTick)&&exactTick.length===2&&Number.isSafeInteger(exactTick[0])&&Number.isSafeInteger(exactTick[1])&&exactTick[1]>0;
    if(!frame||frame.frameIndex!==index||!Number.isSafeInteger(frame.sourcePTS)||!Number.isSafeInteger(frame.projectTick)||!exactTickValid||!Number.isFinite(frame.timestampSeconds)||frame.timestampSeconds<0||(index===0&&frame.timestampSeconds!==0)||(index>0&&frame.timestampSeconds<=previousTimestamp)||(previousPTS!==null&&frame.sourcePTS<=previousPTS)||(previousProjectTick!==null&&frame.projectTick<=previousProjectTick)){{
      throw new Error(`Performance evidence frame ${{index}} has invalid timing`)
    }}
    if(!['observed','missing'].includes(frame.observationState)||frame.semanticState!=='unknown'||frame.neutralityState!=='unknown'){{
      throw new Error(`Performance evidence frame ${{index}} has invalid state`)
    }}
    for(const regionName of requiredEvidenceRegions){{
      const region=frame.regions?.[regionName];
      if(!region||!['observed','missing'].includes(region.observationState)||region.semanticState!=='unknown'||region.neutralityState!=='unknown'){{
        throw new Error(`Performance evidence frame ${{index}} is missing ${{regionName}} state`)
      }}
      if(region.confidence!==null&&(!Number.isFinite(region.confidence)||region.confidence<0||region.confidence>1)){{
        throw new Error(`Performance evidence frame ${{index}} has invalid ${{regionName}} confidence`)
      }}
      if(region.observationState==='missing'&&(region.confidence!==null||region.trackerControls!==null)){{
        throw new Error(`Missing ${{regionName}} evidence must stay null`)
      }}
    }}
    previousTimestamp=frame.timestampSeconds;
    previousPTS=frame.sourcePTS;
    previousProjectTick=frame.projectTick
  }}
  performanceEvidenceSource=payload.source;
  return payload.frames
}}
function validateObservationV3(payload){{
  const requiredQualityStates=['missing','unknown','degraded','diagnostic_clear'];
  if(!payload||payload.schemaVersion!=='autoanim.observation-v3-view/1.0'||payload.sourceMode!=='video_follow'||payload.consumedByRetargeting!==false||payload.claims?.derivedFromVerifiedSealedEvidence!==true||payload.claims?.changesFinalGNMMotion!==false||payload.claims?.confidenceCalibrated!==false||payload.observation?.schemaVersion!=='autoanim.performance-evidence.v3'||payload.observation?.confidenceCalibrated!==false){{
    throw new Error('Unsupported Observation-v3 review contract')
  }}
  const identityTransform=[[1,0,0],[0,1,0],[0,0,1]],display=payload.display,binding=payload.evidenceBinding,displayTimes=display?.frameTimestampsSeconds,hexSha=/^[0-9a-f]{{64}}$/;
  const displayTransformValid=Array.isArray(display?.sourceToDisplayPixelTransform)&&display.sourceToDisplayPixelTransform.length===3&&display.sourceToDisplayPixelTransform.every((row,index)=>Array.isArray(row)&&row.length===3&&row.every((value,column)=>value===identityTransform[index][column]));
  const displayTimesValid=Array.isArray(displayTimes)&&displayTimes.length===payload.source?.frameCount&&displayTimes.every((value,index)=>Number.isFinite(value)&&value>=0&&(index===0?value===0:value>displayTimes[index-1]));
  if(!Array.isArray(payload.frames)||payload.frames.length!==evidenceFrames.length||payload.source?.frameCount!==payload.frames.length||!Array.isArray(payload.source?.frameSize)||payload.source.frameSize.length!==2||!payload.source.frameSize.every(value=>Number.isSafeInteger(value)&&value>0)||payload.source?.sha256!==performanceEvidenceSource?.sha256||binding?.chainVerified!==true||!hexSha.test(binding?.manifestSha256)||!displayTransformValid||!displayTimesValid||display?.clockVerified!==true||display?.frameCount!==payload.source.frameCount||display?.displayRotationDegrees!==0||display?.sampleAspectRatio?.[0]!==1||display?.sampleAspectRatio?.[1]!==1||!Array.isArray(display?.cleanApertureCropLTRB)||display.cleanApertureCropLTRB.some(value=>value!==0)||!hexSha.test(display?.artifact?.sha256)||display?.artifact?.logicalName!=='viewer_media'||!Number.isSafeInteger(display?.artifact?.bytes)||display.artifact.bytes<=0||!Number.isFinite(display?.timestampMaxErrorSeconds)||display.timestampMaxErrorSeconds<0||display.timestampMaxErrorSeconds>.002||!Array.isArray(display?.frameSize)||display.frameSize.some((value,index)=>value!==payload.source.frameSize[index])){{
    throw new Error('Observation-v3 review source does not match performance evidence')
  }}
  const declaredRegions=payload.observation?.regionOrder,declaredReasons=payload.observation?.reasonCodes;
  if(!Array.isArray(declaredRegions)||declaredRegions.length!==requiredEvidenceRegions.length||!requiredEvidenceRegions.every((name,index)=>declaredRegions[index]===name)||!Array.isArray(declaredReasons)||!declaredReasons.every(value=>typeof value==='string')){{
    throw new Error('Observation-v3 review declarations are invalid')
  }}
  for(let index=0;index<payload.frames.length;index++){{
    const frame=payload.frames[index],evidence=evidenceFrames[index];
    if(!frame||frame.frameIndex!==index||frame.sourcePTS!==evidence.sourcePTS||!Number.isFinite(frame.timestampSeconds)||Math.abs(frame.timestampSeconds-evidence.timestampSeconds)>1e-7||Math.abs(displayTimes[index]-frame.timestampSeconds)>.002||typeof frame.detected!=='boolean'||typeof frame.photometricDiscontinuityCandidate!=='boolean'||typeof frame.cutCandidate!=='boolean'||typeof frame.observationEpochStart!=='boolean'){{
      throw new Error(`Observation-v3 frame ${{index}} does not match the source clock`)
    }}
    for(const regionName of requiredEvidenceRegions){{
      const region=frame.regions?.[regionName];
      const box=region?.roiBoxXYXY,boxValid=box===null||(Array.isArray(box)&&box.length===4&&box.every(Number.isSafeInteger)&&box[0]>=0&&box[1]>=0&&box[2]>box[0]&&box[3]>box[1]&&box[2]<=payload.source.frameSize[0]&&box[3]<=payload.source.frameSize[1]);
      const roiAvailable=box!==null,unitValue=value=>Number.isFinite(value)&&value>=0&&value<=1,staticMetrics=['clippedFraction','focusMetric','focusScore','lumaMean','shadowFraction','highlightFraction','dynamicRange'],staticMetricsValid=roiAvailable?staticMetrics.every(name=>unitValue(region?.[name])):staticMetrics.every(name=>region?.[name]===null),pixelCountValid=roiAvailable?Number.isSafeInteger(region?.roiPixelCount)&&region.roiPixelCount===(box[2]-box[0])*(box[3]-box[1]):region?.roiPixelCount===0,previousBox=index>0?payload.frames[index-1]?.regions?.[regionName]?.roiBoxXYXY:null,temporalExpected=roiAvailable&&Array.isArray(previousBox)&&!frame.observationEpochStart,temporalValid=temporalExpected?unitValue(region?.temporalInnovation):region?.temporalInnovation===null,flowValid=temporalExpected?(region?.flowConsistency===null||unitValue(region.flowConsistency)):region?.flowConsistency===null;
      if(!region||!requiredQualityStates.includes(region.qualityState)||region.confidenceCalibrated!==false||region.occlusionState!=='unknown'||region.identityContinuityState!=='unknown'||!Array.isArray(region.reasonCodes)||!region.reasonCodes.every(value=>declaredReasons.includes(value))||!Number.isFinite(payload.observation.confidenceCap)||payload.observation.confidenceCap<=0||payload.observation.confidenceCap>=.75||!(region.confidence===null||(Number.isFinite(region.confidence)&&region.confidence>=0&&region.confidence<=payload.observation.confidenceCap))||!boxValid||!pixelCountValid||!staticMetricsValid||!temporalValid||!flowValid||(roiAvailable?region.confidence===null:region.confidence!==null)){{
        throw new Error(`Observation-v3 frame ${{index}} has invalid ${{regionName}} diagnostics`)
      }}
    }}
  }}
  diagnosticSource=payload.source;
  diagnosticDisplay=display;
  return payload.frames
}}
function confidenceText(region){{
  if(!region||region.observationState==='missing')return 'MISSING · confidence —';
  const confidence=region.confidence===null?'—':`${{Math.round(region.confidence*100)}}%`;
  return `${{region.semanticState==='unknown'?'UNKNOWN':'OBSERVED'}} · ${{confidence}}`
}}
const diagnosticRegionColors={{mouth:'#ff6f91',eyes:'#69b9ff',upperFace:'#d8ff63',head:'#ffbd69'}};
function diagnosticRegionText(regionName,trackerRegion,diagnosticFrame){{
  const tracker=`TRACKER ${{confidenceText(trackerRegion)}}`;
  if(!diagnosticsReady||!diagnosticFrame)return `${{tracker}}\nPIXELS unavailable`;
  const region=diagnosticFrame.regions[regionName],confidence=region.confidence===null?'—':region.confidence.toFixed(2);
  const reasons=region.reasonCodes.length?` · ${{region.reasonCodes.slice(0,2).join(', ')}}${{region.reasonCodes.length>2?` +${{region.reasonCodes.length-2}}`:''}}`:'';
  return `${{tracker}}\nPIXELS ${{region.qualityState.toUpperCase()}} · uncalibrated score ${{confidence}}${{reasons}}`
}}
function drawDiagnosticOverlay(diagnosticFrame){{
  if(!diagnosticOverlay||!media||!diagnosticsReady||!diagnosticFrame||!diagnosticSource)return;
  const bounds=media.getBoundingClientRect(),width=Math.max(bounds.width,1),height=Math.max(bounds.height,1),ratio=Math.min(devicePixelRatio||1,2);
  const backingWidth=Math.round(width*ratio),backingHeight=Math.round(height*ratio);if(diagnosticOverlay.width!==backingWidth)diagnosticOverlay.width=backingWidth;if(diagnosticOverlay.height!==backingHeight)diagnosticOverlay.height=backingHeight;
  const context=diagnosticOverlay.getContext('2d');context.setTransform(ratio,0,0,ratio,0,0);context.clearRect(0,0,width,height);
  const sourceWidth=diagnosticSource.frameSize[0],sourceHeight=diagnosticSource.frameSize[1],scale=Math.min(width/sourceWidth,height/sourceHeight),offsetX=(width-sourceWidth*scale)/2,offsetY=(height-sourceHeight*scale)/2;
  {bridge_selected_region_declaration}
  context.font='10px ui-monospace,monospace';context.lineWidth=1.5;
  for(const regionName of requiredEvidenceRegions){{
    const region=diagnosticFrame.regions[regionName],box=region.roiBoxXYXY;
    if(!box)continue;
    const x=offsetX+box[0]*scale,y=offsetY+box[1]*scale,w=(box[2]-box[0])*scale,h=(box[3]-box[1])*scale;
    {bridge_region_alpha_hook}context.strokeStyle=diagnosticRegionColors[regionName];context.setLineDash(region.qualityState==='diagnostic_clear'?[]:[4,3]);context.strokeRect(x,y,w,h);
    const label=`${{regionName}} score ${{region.confidence===null?'—':region.confidence.toFixed(2)}}`,labelWidth=context.measureText(label).width+6,labelY=Math.max(0,y-14);
    context.fillStyle='#080a0cdd';context.fillRect(x,labelY,labelWidth,14);context.fillStyle=diagnosticRegionColors[regionName];context.fillText(label,x+3,labelY+10)
  }}
  {bridge_region_alpha_reset}context.setLineDash([])
}}
function evidenceIndexAtOrBefore(time){{
  if(!evidenceFrames.length)return-1;
  let low=0,high=evidenceFrames.length-1,result=0;
  while(low<=high){{
    const middle=(low+high)>>1;
    if(evidenceFrames[middle].timestampSeconds<=time+1e-7){{result=middle;low=middle+1}}
    else high=middle-1
  }}
  return result
}}
function evidenceIndexForPresentedMediaTime(time){{
  const times=diagnosticDisplay?.frameTimestampsSeconds;
  if(!Array.isArray(times)||times.length!==evidenceFrames.length)return evidenceIndexAtOrBefore(time);
  let low=0,high=times.length-1;
  while(low<=high){{const middle=(low+high)>>1;if(times[middle]<time)low=middle+1;else high=middle-1}}
  if(low<=0)return 0;if(low>=times.length)return times.length-1;
  return Math.abs(times[low]-time)<Math.abs(time-times[low-1])?low:low-1
}}
function refreshEvidenceControls(){{
  const enabled=evidenceReady&&media&&media.readyState>=1&&pendingEvidenceFrameIndex<0&&(!reviewFrameUrlTemplate||diagnosticsReady);
  previousSourceFrame.disabled=!enabled||displayedEvidenceFrameIndex<=0;
  nextSourceFrame.disabled=!enabled||displayedEvidenceFrameIndex>=evidenceFrames.length-1
}}
function showEvidenceFrame(index){{
  if(!evidenceFrames.length)return;
  const bounded=Math.min(Math.max(index,0),evidenceFrames.length-1),frame=evidenceFrames[bounded],diagnosticFrame=diagnosticsReady?diagnosticFrames[bounded]:null;
  if(displayedEvidenceFrameIndex===bounded&&displayedDiagnosticsReady===diagnosticsReady){{refreshEvidenceControls();return}}
  displayedEvidenceFrameIndex=bounded;
  displayedDiagnosticsReady=diagnosticsReady;
  const clockLabel=presentationClockState==='verified_static'?'server-decoded proxy frame verified':presentationClockState==='verified'?'presented frame verified':presentationClockState==='fallback'?'presented-frame callback unavailable':'awaiting presented frame';
  evidenceReadout.textContent=`Frame ${{frame.frameIndex+1}}/${{evidenceFrames.length}} · PTS ${{frame.sourcePTS}} · tick ${{frame.projectTick}} · ${{frame.timestampSeconds.toFixed(3)}} s · ${{clockLabel}}`;
  if(frame.observationState==='missing'){{
    evidenceFlag.className='missing';
    evidenceFlag.textContent='MISSING · no face observation at this source frame'
  }}else{{
    evidenceFlag.className='unknown';
    evidenceFlag.textContent='UNKNOWN · observed tracker values are not a labeled neutral or expression state'
  }}
  for(const regionName of requiredEvidenceRegions){{
    evidenceRegionElements[regionName].textContent=diagnosticRegionText(regionName,frame.regions[regionName],diagnosticFrame)
  }}
  if(diagnosticFrame){{
    const events=[];if(diagnosticFrame.observationEpochStart)events.push('EPOCH START');if(diagnosticFrame.photometricDiscontinuityCandidate)events.push('PHOTOMETRIC CANDIDATE');if(diagnosticFrame.cutCandidate)events.push('CUT CANDIDATE');
    evidenceDiagnosticState.textContent=`Observation-v3 · provisional diagnostic only · never motion authority${{events.length?' · '+events.join(' · '):''}}`;
    drawDiagnosticOverlay(diagnosticFrame)
  }}else{{evidenceDiagnosticState.textContent='Pixel diagnostics unavailable · tracker evidence only';if(diagnosticOverlay){{const context=diagnosticOverlay.getContext('2d');context.clearRect(0,0,diagnosticOverlay.width,diagnosticOverlay.height)}}}}
  {bridge_cursor_hook}
  refreshEvidenceControls()
}}
function presentedFrameTolerance(index){{
  const times=diagnosticDisplay?.frameTimestampsSeconds||evidenceFrames.map(frame=>frame.timestampSeconds),previous=index>0?times[index]-times[index-1]:Infinity,next=index+1<times.length?times[index+1]-times[index]:Infinity,spacing=Math.min(previous,next);
  return Math.min(Number.isFinite(spacing)?spacing*.2:.004,.004)
}}
function applyPresentedMediaTime(mediaTime,verified){{
  if(!Number.isFinite(mediaTime)||mediaTime<0)return 'invalid';
  presentedMediaTime=mediaTime;
  presentationClockState=verified&&diagnosticDisplay?'verified':'fallback';
  if(!evidenceReady)return 'none';
  const actualIndex=evidenceIndexForPresentedMediaTime(mediaTime);
  let pendingResult='none';
  if(pendingEvidenceFrameIndex>=0){{
    const expected=evidenceFrames[pendingEvidenceFrameIndex];
    const expectedMediaTime=diagnosticDisplay?.frameTimestampsSeconds?.[pendingEvidenceFrameIndex]??expected.timestampSeconds;
    if(actualIndex!==pendingEvidenceFrameIndex||Math.abs(mediaTime-expectedMediaTime)>presentedFrameTolerance(pendingEvidenceFrameIndex)){{
      evidenceError.hidden=false;
      evidenceError.textContent=`Requested frame ${{pendingEvidenceFrameIndex+1}} but the browser presented frame ${{actualIndex+1}}; retrying while diagnostics follow the presented frame.`;
      pendingResult='mismatch'
    }}else{{
      evidenceError.hidden=true;
      evidenceError.textContent='';pendingEvidenceFrameIndex=-1;pendingPresentationAttempt=0;if(pendingPresentationTimer!==null){{clearTimeout(pendingPresentationTimer);pendingPresentationTimer=null}};pendingResult='matched'
    }}
  }}
  showEvidenceFrame(actualIndex);
  if(mixer&&animationAction&&media.paused){{
    const exactTime=evidenceFrames[actualIndex].timestampSeconds;
    animationAction.time=Math.min(Math.max(exactTime,0),animationDuration);mixer.update(0)
  }}
  return pendingResult
}}
function presentedVideoFrameLoop(_now,metadata){{
  if(staticReviewFrameIndex>=0){{videoFrameCallbackId=media.requestVideoFrameCallback(presentedVideoFrameLoop);return}}
  const forcedPresentation=forcePauseAfterPresentation;
  const mediaTime=Number(metadata?.mediaTime),presentedIndex=evidenceReady?evidenceIndexForPresentedMediaTime(mediaTime):-1,expectedMediaTime=pendingEvidenceFrameIndex>=0?(diagnosticDisplay?.frameTimestampsSeconds?.[pendingEvidenceFrameIndex]??evidenceFrames[pendingEvidenceFrameIndex].timestampSeconds):null,presentsTarget=forcedPresentation&&pendingEvidenceFrameIndex>=0&&presentedIndex===pendingEvidenceFrameIndex&&Math.abs(mediaTime-expectedMediaTime)<=presentedFrameTolerance(pendingEvidenceFrameIndex);
  if(presentsTarget){{media.pause();media.muted=mutedBeforeForcedPresentation;forcePauseAfterPresentation=false}}
  const result=applyPresentedMediaTime(mediaTime,true);
  if(forcedPresentation&&result==='mismatch'&&pendingEvidenceFrameIndex>=0){{
    pendingPresentationAttempt+=1;
    if(pendingPresentationAttempt>3)failPendingPresentation('The browser advanced past the requested source frame without presenting it.');
  }}
  videoFrameCallbackId=media.requestVideoFrameCallback(presentedVideoFrameLoop)
}}
function startPresentedVideoFrameClock(){{
  if(!supportsPresentedFrameClock||videoFrameCallbackId!==null)return;
  videoFrameCallbackId=media.requestVideoFrameCallback(presentedVideoFrameLoop)
}}
function failPendingPresentation(message){{
  if(pendingPresentationTimer!==null){{clearTimeout(pendingPresentationTimer);pendingPresentationTimer=null}}
  if(forcePauseAfterPresentation){{media.pause();media.muted=mutedBeforeForcedPresentation;forcePauseAfterPresentation=false}}
  pendingEvidenceFrameIndex=-1;pendingPresentationAttempt=0;presentationClockState='waiting';evidenceError.hidden=false;evidenceError.textContent=message;refreshEvidenceControls()
}}
function armPendingPresentationTimeout(){{
  if(pendingPresentationTimer!==null)clearTimeout(pendingPresentationTimer);
  pendingPresentationTimer=setTimeout(()=>failPendingPresentation('Timed out waiting for the browser to present the requested source frame.'),2000)
}}
async function forcePausedSeekPresentation(){{
  if(!supportsPresentedFrameClock||forcePauseAfterPresentation||!media.paused)return;
  mutedBeforeForcedPresentation=media.muted;media.muted=true;forcePauseAfterPresentation=true;
  try{{await media.play()}}catch(error){{
    forcePauseAfterPresentation=false;media.muted=mutedBeforeForcedPresentation;failPendingPresentation(`The browser could not present the requested paused frame: ${{error.message}}`)
  }}
}}
function seekPendingEvidenceFrame(){{
  if(pendingEvidenceFrameIndex<0||!media)return;
  const targetTime=diagnosticDisplay?.frameTimestampsSeconds?.[pendingEvidenceFrameIndex]??evidenceFrames[pendingEvidenceFrameIndex].timestampSeconds;
  media.pause();armPendingPresentationTimeout();
  if(Math.abs(Number(media.currentTime)-targetTime)<=1e-7)void forcePausedSeekPresentation();
  else media.currentTime=targetTime
}}
function clearStaticReviewFrame(){{
  staticReviewFrameIndex=-1;staticReviewInternalSeekTime=null;if(reviewFrameImage)reviewFrameImage.hidden=true;
  if(reviewFrameObjectUrl!==null){{URL.revokeObjectURL(reviewFrameObjectUrl);reviewFrameObjectUrl=null}}
}}
async function displayExactReviewFrame(index){{
  reviewFrameAbortController?.abort();const controller=new AbortController();reviewFrameAbortController=controller;
  const timeout=setTimeout(()=>controller.abort(),5000);
  try{{
    const response=await fetch(reviewFrameUrl(index),{{credentials:'same-origin',cache:'no-store',headers:{{Accept:'image/png'}},signal:controller.signal}});
    if(!response.ok)throw new Error(`Exact frame request failed (${{response.status}})`);
    if(response.headers.get('x-autoanim-frame-index')!==String(index)||response.headers.get('x-autoanim-proxy-sha256')!==diagnosticDisplay?.artifact?.sha256)throw new Error('Exact frame response is not bound to the reviewed proxy');
    const length=Number(response.headers.get('content-length')||0);if(length>64*1024*1024)throw new Error('Exact frame response is too large');
    const blob=await response.blob();if(blob.type!=='image/png'||blob.size<=0||blob.size>64*1024*1024)throw new Error('Exact frame response is not a bounded PNG');
    const objectUrl=URL.createObjectURL(blob);reviewFrameImage.src=objectUrl;await reviewFrameImage.decode();
    if(pendingEvidenceFrameIndex!==index){{URL.revokeObjectURL(objectUrl);return}}
    clearStaticReviewFrame();reviewFrameObjectUrl=objectUrl;reviewFrameImage.src=objectUrl;reviewFrameImage.hidden=false;staticReviewFrameIndex=index;
    presentedMediaTime=diagnosticDisplay.frameTimestampsSeconds[index];presentationClockState='verified_static';pendingEvidenceFrameIndex=-1;pendingPresentationAttempt=0;if(pendingPresentationTimer!==null){{clearTimeout(pendingPresentationTimer);pendingPresentationTimer=null}};
    evidenceError.hidden=true;evidenceError.textContent='';displayedEvidenceFrameIndex=-1;showEvidenceFrame(index);staticReviewInternalSeekTime=presentedMediaTime;media.currentTime=presentedMediaTime;
    if(mixer&&animationAction){{const exactTime=evidenceFrames[index].timestampSeconds;animationAction.time=Math.min(Math.max(exactTime,0),animationDuration);mixer.update(0)}}
    {bridge_exact_frame_hook}
  }}catch(error){{if(error.name!=='AbortError'||pendingEvidenceFrameIndex===index){{failPendingPresentation(`Exact review frame unavailable: ${{error.message}}`);{bridge_exact_frame_error_hook}}}}}
  finally{{clearTimeout(timeout);if(reviewFrameAbortController===controller)reviewFrameAbortController=null}}
}}
function stepEvidenceFrame(direction){{
  if(!evidenceReady||!media)return;
  const current=displayedEvidenceFrameIndex>=0?displayedEvidenceFrameIndex:evidenceIndexAtOrBefore(presentedMediaTime??Number(media.currentTime));
  const target=Math.min(Math.max(current+(direction<0?-1:1),0),evidenceFrames.length-1);
  media.pause();
  pendingEvidenceFrameIndex=target;pendingPresentationAttempt=0;
  presentationClockState='waiting';
  evidenceReadout.textContent=`Seeking source frame ${{target+1}}/${{evidenceFrames.length}} · waiting for browser presentation…`;
  refreshEvidenceControls();
  if(reviewFrameUrlTemplate&&diagnosticsReady)void displayExactReviewFrame(target);else seekPendingEvidenceFrame();
}}
async function loadPerformanceEvidence(){{
  if(!performanceEvidenceUrl||!media||mediaKind!=='video')return;
  evidencePanel.hidden=false;
  evidenceReadout.textContent='Loading source-frame evidence…';
  try{{
    const response=await fetch(validateEvidenceArtifactUrl(performanceEvidenceUrl),{{credentials:'same-origin',cache:'no-store',headers:{{Accept:'application/json'}},signal:evidenceAbortController.signal}});
    if(!response.ok)throw new Error(`Evidence request failed (${{response.status}})`);
    evidenceFrames=validatePerformanceEvidence(await response.json());
    evidenceReady=true;
    if(supportsPresentedFrameClock){{startPresentedVideoFrameClock();if(presentedMediaTime!==null)applyPresentedMediaTime(presentedMediaTime,true);else evidenceReadout.textContent='Waiting for the first presented video frame…'}}
    else applyPresentedMediaTime(Number(media.currentTime),false);
    refreshEvidenceControls();
    if(observationV3Url)await loadObservationV3Diagnostics();
    {bridge_evidence_hook}
  }}catch(error){{
    evidenceError.hidden=false;
    evidenceError.textContent=`Evidence unavailable: ${{error.message}}`;
    evidenceReadout.textContent='Source-frame evidence unavailable';
    evidenceFlag.className='missing';{bridge_evidence_error_hook}
    evidenceFlag.textContent='MISSING · diagnostic artifact was not accepted'
  }}
}}
async function loadObservationV3Diagnostics(){{
  try{{
    const response=await fetch(validateObservationV3Url(observationV3Url),{{credentials:'same-origin',cache:'no-store',headers:{{Accept:'application/json'}},signal:evidenceAbortController.signal}});
    if(!response.ok)throw new Error(`Observation-v3 request failed (${{response.status}})`);
    diagnosticFrames=validateObservationV3(await response.json());diagnosticsReady=true;displayedEvidenceFrameIndex=-1;if(presentedMediaTime!==null)applyPresentedMediaTime(presentedMediaTime,supportsPresentedFrameClock)
  }}catch(error){{
    diagnosticsReady=false;diagnosticFrames=[];evidenceError.hidden=false;evidenceError.textContent=`Pixel diagnostics unavailable: ${{error.message}}`;evidenceDiagnosticState.textContent='Pixel diagnostics rejected · tracker evidence remains available';{bridge_diagnostics_error_hook}
  }}
}}
previousSourceFrame.addEventListener('click',()=>stepEvidenceFrame(-1));
nextSourceFrame.addEventListener('click',()=>stepEvidenceFrame(1));
if(media&&performanceEvidenceUrl&&mediaKind==='video'){{
  media.addEventListener('loadedmetadata',()=>{{refreshEvidenceControls();startPresentedVideoFrameClock();if(diagnosticsReady&&displayedEvidenceFrameIndex>=0)drawDiagnosticOverlay(diagnosticFrames[displayedEvidenceFrameIndex])}});
  media.addEventListener('play',()=>{{if(forcePauseAfterPresentation)return;if(reviewFrameUrlTemplate&&pendingEvidenceFrameIndex>=0){{reviewFrameAbortController?.abort();reviewFrameAbortController=null;pendingEvidenceFrameIndex=-1;pendingPresentationAttempt=0;if(pendingPresentationTimer!==null){{clearTimeout(pendingPresentationTimer);pendingPresentationTimer=null}};evidenceError.hidden=true;evidenceError.textContent='';refreshEvidenceControls()}}if(staticReviewFrameIndex>=0)clearStaticReviewFrame();presentationClockState='waiting'}});
  media.addEventListener('seeking',()=>{{if(staticReviewFrameIndex>=0&&staticReviewInternalSeekTime===null){{clearStaticReviewFrame();presentationClockState='waiting'}}}});
  media.addEventListener('seeked',()=>{{if(staticReviewFrameIndex>=0&&staticReviewInternalSeekTime!==null&&Math.abs(Number(media.currentTime)-staticReviewInternalSeekTime)<=presentedFrameTolerance(staticReviewFrameIndex)){{staticReviewInternalSeekTime=null;return}}if(staticReviewFrameIndex>=0){{clearStaticReviewFrame();presentationClockState='waiting'}}if(reviewFrameUrlTemplate&&diagnosticsReady&&media.paused&&evidenceReady){{const soughtIndex=evidenceIndexForPresentedMediaTime(Number(media.currentTime));pendingEvidenceFrameIndex=soughtIndex;pendingPresentationAttempt=0;void displayExactReviewFrame(soughtIndex);return}}if(supportsPresentedFrameClock){{if(pendingEvidenceFrameIndex<0&&evidenceReady){{const soughtIndex=evidenceIndexForPresentedMediaTime(Number(media.currentTime)),expectedTime=diagnosticDisplay?.frameTimestampsSeconds?.[soughtIndex]??evidenceFrames[soughtIndex].timestampSeconds,alreadyPresented=presentedMediaTime!==null&&Math.abs(presentedMediaTime-expectedTime)<=presentedFrameTolerance(soughtIndex);if(!alreadyPresented)pendingEvidenceFrameIndex=soughtIndex}}if(pendingEvidenceFrameIndex>=0)void forcePausedSeekPresentation()}}else applyPresentedMediaTime(Number(media.currentTime),false)}});
  media.addEventListener('timeupdate',()=>{{if(!supportsPresentedFrameClock)applyPresentedMediaTime(Number(media.currentTime),false)}});
  void loadPerformanceEvidence()
}}
let mediaResizeObserver=null;if(mediaShell){{mediaResizeObserver=new ResizeObserver(()=>{{if(diagnosticsReady&&displayedEvidenceFrameIndex>=0)drawDiagnosticOverlay(diagnosticFrames[displayedEvidenceFrameIndex])}});mediaResizeObserver.observe(mediaShell)}}
function cloneMaterials(material,wireOnly=false){{const list=Array.isArray(material)?material:[material];const cloned=list.map(source=>{{if(wireOnly)return new THREE.MeshBasicMaterial({{color:0xd8ff63,wireframe:true,transparent:true,opacity:.78}});const copy=source.clone();copy.wireframe=false;return copy}});return Array.isArray(material)?cloned:cloned[0]}}
function applyMode(){{if(!root)return;const mode=document.querySelector('#mode').value;root.traverse(node=>{{if(!node.isMesh)return;if(mode==='wire')node.material=node.userData.wireMaterial;else{{node.material=node.userData.surfaceMaterial;const mats=Array.isArray(node.material)?node.material:[node.material];mats.forEach(material=>material.wireframe=mode==='surface-wire')}}}})}}
function frameObject(object){{const box=new THREE.Box3().setFromObject(object),center=box.getCenter(new THREE.Vector3()),size=box.getSize(new THREE.Vector3()),radius=Math.max(size.x,size.y,size.z);homeTarget.copy(center);homePosition.set(center.x,center.y,center.z+radius*2.7);camera.near=Math.max(radius/500,.001);camera.far=Math.max(radius*20,2);camera.updateProjectionMatrix();camera.position.copy(homePosition);controls.minDistance=radius*1.2;controls.maxDistance=radius*8;controls.target.copy(homeTarget);controls.update();grid.position.set(center.x,box.min.y-.002,center.z)}}
new GLTFLoader().load(assetUrl,gltf=>{{root=gltf.scene;let vertices=0,triangles=0,meshes=0;root.traverse(node=>{{if(node.isMesh){{meshes++;const geometry=node.geometry;vertices+=geometry.getAttribute('position').count;triangles+=(geometry.index?geometry.index.count:geometry.getAttribute('position').count)/3;node.userData.surfaceMaterial=cloneMaterials(node.material);node.userData.wireMaterial=cloneMaterials(node.material,true)}}}});scene.add(root);if(gltf.animations.length){{const clip=gltf.animations[0];mixer=new THREE.AnimationMixer(root);animationAction=mixer.clipAction(clip);animationDuration=clip.duration;animationAction.setLoop(THREE.LoopOnce,1);animationAction.clampWhenFinished=true;animationAction.play();animationAction.paused=true;animationAction.time=0;mixer.update(0)}}frameObject(root);applyMode();document.querySelector('#metrics').textContent=`${{vertices.toLocaleString()}} render vertices · ${{Math.round(triangles).toLocaleString()}} triangles · ${{meshes}} primitive${{meshes===1?'':'s'}} · ${{gltf.animations.length?'animated':'static'}}`;status.textContent=mixer&&media?'Ready · media controls drive exact 3D time':'Ready · drag to orbit · scroll to zoom';{bridge_renderer_ready_hook}}},undefined,error=>{{status.textContent=`Could not load 3D asset: ${{error.message}}`;status.style.color='#ff7d7d';{bridge_renderer_error_hook}}});
document.querySelector('#mode').addEventListener('change',{mode_change_listener});document.querySelector('#exposure').addEventListener('input',event=>renderer.toneMappingExposure=Number(event.target.value));document.querySelector('#reset').addEventListener('click',()=>{{camera.position.copy(homePosition);controls.target.copy(homeTarget);controls.update()}});
// Sample the paused action directly: AnimationMixer.setTime() zeroes action-local
// time, so a LoopOnce action that has reached the end cannot seek backward correctly.
function refreshPlaybackStatus(){{
  if(!mixer||!animationAction||!media)return;
  const playbackTime=mediaKind==='video'&&presentedMediaTime!==null?presentedMediaTime:media.currentTime,time=playbackTime.toFixed(2),clockLabel=mediaKind==='video'?(presentationClockState==='verified_static'?'server-decoded frame synchronized':presentationClockState==='verified'?'presented-frame synchronized':'media-clock fallback'):'media-clock synchronized';
  if(!media.paused&&!media.ended)status.textContent=`Playing ${{time}} s · ${{clockLabel}}`;
  else if(media.ended)status.textContent=`Finished ${{time}} s · ${{clockLabel}}`;
  else if(media.currentTime<=.01)status.textContent='Ready · media controls drive exact 3D time';
  else status.textContent=`Paused ${{time}} s · ${{clockLabel}}`
}}
function resize(){{const width=stage.clientWidth,height=stage.clientHeight;renderer.setSize(width,height,false);camera.aspect=width/Math.max(height,1);camera.updateProjectionMatrix()}}const stageResizeObserver=new ResizeObserver(resize);stageResizeObserver.observe(stage);resize();renderer.setAnimationLoop(()=>{{if(mixer&&animationAction&&media){{const clockTime=mediaKind==='video'&&presentedMediaTime!==null?presentedMediaTime:media.currentTime;animationAction.time=Math.min(Math.max(clockTime,0),animationDuration);mixer.update(0);refreshPlaybackStatus()}}controls.update();renderer.render(scene,camera)}});
window.addEventListener('pagehide',()=>{{{bridge_shutdown_hook}evidenceAbortController.abort();reviewFrameAbortController?.abort();clearStaticReviewFrame();if(pendingPresentationTimer!==null)clearTimeout(pendingPresentationTimer);if(forcePauseAfterPresentation&&media){{media.pause();media.muted=mutedBeforeForcedPresentation}}if(videoFrameCallbackId!==null&&typeof media?.cancelVideoFrameCallback==='function')media.cancelVideoFrameCallback(videoFrameCallbackId);renderer.setAnimationLoop(null);stageResizeObserver.disconnect();mediaResizeObserver?.disconnect();controls.dispose();if(mixer)mixer.stopAllAction();if(root)root.traverse(node=>{{if(!node.isMesh)return;node.geometry?.dispose();const materials=[].concat(node.material||[],node.userData.surfaceMaterial||[],node.userData.wireMaterial||[]);for(const material of new Set(materials)){{material.map?.dispose();material.dispose?.()}}}});renderer.dispose()}});
</script></body></html>"""
