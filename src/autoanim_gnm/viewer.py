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


def viewer_html(
    *,
    asset_url: str,
    title: str,
    media_url: str | None = None,
    media_type: str | None = None,
    performance_evidence_url: str | None = None,
    metadata: dict | None = None,
    vendor_base_url: str = f"/api/viewer/vendor/{VIEWER_THREE_VERSION}",
) -> str:
    """Return a self-contained shell for allowlisted GLB, media, and evidence."""

    safe_title = html.escape(title)
    encoded_url = _script_json(asset_url)
    encoded_media_url = _script_json(media_url)
    encoded_media_kind = _script_json(
        "video" if media_type is not None and media_type.startswith("video/") else "audio"
    )
    encoded_evidence_url = _script_json(
        _validate_performance_evidence_url(performance_evidence_url)
    )
    encoded_metadata = _script_json(metadata)
    encoded_three_url = _script_json(f"{vendor_base_url}/three.module.js")
    encoded_addons_url = _script_json(f"{vendor_base_url}/addons/")
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
    video{{display:block;max-height:240px;object-fit:contain;padding:0}}
    button{{margin-top:12px;background:var(--accent);color:#111500;border:0;font-weight:800;cursor:pointer}}
    button:disabled{{cursor:not-allowed;opacity:.45}}.evidence-review{{margin-top:18px;padding:12px;border:1px solid var(--line);border-radius:10px;background:#0b0e11}}
    .evidence-review h2{{margin:0 0 8px;font:700 13px/1.2 system-ui,sans-serif}}.frame-nav{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}
    .frame-nav button{{width:100%;margin:0;padding:8px}}#evidence-readout{{margin:9px 0 6px;color:var(--text);font-size:11px}}
    #evidence-flag{{padding:7px 8px;border-radius:7px;background:#252c32;color:var(--muted);font-size:11px}}#evidence-flag.missing{{background:#401f25;color:#ffb3bd}}#evidence-flag.unknown{{background:#322f1b;color:#f3dfa1}}
    #evidence-regions{{display:grid;gap:5px;margin:9px 0 0}}#evidence-regions div{{display:flex;justify-content:space-between;gap:10px;font-size:11px}}#evidence-regions dt{{color:var(--muted)}}#evidence-regions dd{{margin:0;text-align:right}}#evidence-error{{margin:8px 0 0;color:#ff9da8;font-size:11px}}
    #stage{{min-width:0;min-height:0;position:relative}}canvas{{display:block;width:100%;height:100%}}
    #status{{position:absolute;left:18px;bottom:16px;padding:8px 11px;background:#080a0cdd;border:1px solid var(--line);border-radius:8px;color:var(--muted)}}
    .legend{{margin-top:24px;padding-top:16px;border-top:1px solid var(--line);font-size:12px}}#metadata{{white-space:pre-wrap;overflow-wrap:anywhere;color:var(--muted)}}
    @media(max-width:720px){{main{{grid-template-columns:1fr;grid-template-rows:auto 1fr}}aside{{padding:14px;border-right:0;border-bottom:1px solid var(--line)}}aside p,.legend{{display:none}}h1{{font-size:20px}}label{{display:inline-block;margin:6px 6px 4px 0}}select,input,button{{width:auto}}}}
  </style>
  <script type="importmap">{{"imports":{{"three":{encoded_three_url},"three/addons/":{encoded_addons_url}}}}}</script>
</head><body><main>
  <aside><h1>{safe_title}</h1><p>Orbit, zoom, and inspect the exact seam-correct GNM asset exported by this job.</p>
    <label for="mode">Display</label><select id="mode"><option value="surface">Surface / texture</option><option value="surface-wire">Surface + topology</option><option value="wire">Topology only</option></select>
    <label for="exposure">Exposure</label><input id="exposure" type="range" min="0.35" max="2.25" step="0.05" value="1.0">
    <button id="reset" type="button">Reset camera</button><div id="media-slot"></div>
    <section id="evidence-panel" class="evidence-review" aria-label="Performance evidence review" hidden><h2>Source-frame evidence</h2>
      <div class="frame-nav"><button id="previous-source-frame" type="button" disabled>Previous frame</button><button id="next-source-frame" type="button" disabled>Next frame</button></div>
      <div id="evidence-readout" aria-live="polite">Evidence unavailable</div><div id="evidence-flag" class="unknown">UNKNOWN</div>
      <dl id="evidence-regions"><div><dt>Mouth</dt><dd id="evidence-mouth">—</dd></div><div><dt>Eyes</dt><dd id="evidence-eyes">—</dd></div><div><dt>Upper face</dt><dd id="evidence-upper-face">—</dd></div><div><dt>Head</dt><dd id="evidence-head">—</dd></div></dl>
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

const assetUrl={encoded_url},mediaUrl={encoded_media_url},mediaKind={encoded_media_kind},performanceEvidenceUrl={encoded_evidence_url},metadata={encoded_metadata},stage=document.querySelector('#stage'),status=document.querySelector('#status');
if(metadata){{const lines=[];for(const [label,value] of Object.entries(metadata))lines.push(`${{label.replaceAll('_',' ')}}: ${{Array.isArray(value)?value.join(', '):String(value)}}`);document.querySelector('#metadata').textContent=lines.join('\\n')}}
const scene=new THREE.Scene();scene.background=new THREE.Color(0x0b0e10);
const camera=new THREE.PerspectiveCamera(28,1,.005,20);
let renderer;try{{renderer=new THREE.WebGLRenderer({{antialias:true,alpha:false}})}}catch(error){{status.textContent='WebGL is unavailable. Download the GLB from the job result to inspect it in another viewer.';status.style.color='#ff7d7d';throw error}}renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1;renderer.domElement.setAttribute('role','img');renderer.domElement.setAttribute('aria-label','Interactive 3D GNM head');renderer.domElement.tabIndex=0;stage.prepend(renderer.domElement);
const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.dampingFactor=.075;controls.screenSpacePanning=true;controls.enablePan=false;controls.minPolarAngle=.2;controls.maxPolarAngle=Math.PI-.2;
scene.add(new THREE.HemisphereLight(0xeaf3ff,0x33291f,2.2));const key=new THREE.DirectionalLight(0xffffff,3.1);key.position.set(.8,.7,1.4);scene.add(key);const rim=new THREE.DirectionalLight(0xb7d8ff,1.8);rim.position.set(-1,.35,-1);scene.add(rim);
const grid=new THREE.GridHelper(1,20,0x4c555d,0x252b30);scene.add(grid);
let root=null,mixer=null,animationAction=null,animationDuration=0,homePosition=new THREE.Vector3(),homeTarget=new THREE.Vector3();
let media=null;if(mediaUrl){{const label=document.createElement('label');label.textContent=mediaKind==='video'?'Source performance':'Playback';media=document.createElement(mediaKind);media.controls=true;media.preload='metadata';if(mediaKind==='video')media.playsInline=true;media.src=mediaUrl;document.querySelector('#media-slot').append(label,media)}}
const evidencePanel=document.querySelector('#evidence-panel');
const evidenceReadout=document.querySelector('#evidence-readout');
const evidenceFlag=document.querySelector('#evidence-flag');
const evidenceError=document.querySelector('#evidence-error');
const previousSourceFrame=document.querySelector('#previous-source-frame');
const nextSourceFrame=document.querySelector('#next-source-frame');
const evidenceRegionElements={{
  mouth:document.querySelector('#evidence-mouth'),
  eyes:document.querySelector('#evidence-eyes'),
  upperFace:document.querySelector('#evidence-upper-face'),
  head:document.querySelector('#evidence-head'),
}};
const requiredEvidenceRegions=['mouth','eyes','upperFace','head'];
let evidenceFrames=[],displayedEvidenceFrameIndex=-1,evidenceReady=false;
function validateEvidenceArtifactUrl(value){{
  const parsed=new URL(value,window.location.href);
  const allowlisted=/^\\/api\\/jobs\\/[A-Za-z0-9_-]+\\/files\\/performance-evidence\\.json$/;
  if(parsed.origin!==window.location.origin||!allowlisted.test(parsed.pathname)||parsed.search||parsed.hash){{
    throw new Error('Evidence URL is not an allowlisted same-origin job artifact')
  }}
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
  return payload.frames
}}
function confidenceText(region){{
  if(!region||region.observationState==='missing')return 'MISSING · confidence —';
  const confidence=region.confidence===null?'—':`${{Math.round(region.confidence*100)}}%`;
  return `${{region.semanticState==='unknown'?'UNKNOWN':'OBSERVED'}} · ${{confidence}}`
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
function refreshEvidenceControls(){{
  const enabled=evidenceReady&&media&&media.readyState>=1;
  previousSourceFrame.disabled=!enabled||displayedEvidenceFrameIndex<=0;
  nextSourceFrame.disabled=!enabled||displayedEvidenceFrameIndex>=evidenceFrames.length-1
}}
function showEvidenceFrame(index){{
  if(!evidenceFrames.length)return;
  const bounded=Math.min(Math.max(index,0),evidenceFrames.length-1),frame=evidenceFrames[bounded];
  if(displayedEvidenceFrameIndex===bounded){{refreshEvidenceControls();return}}
  displayedEvidenceFrameIndex=bounded;
  evidenceReadout.textContent=`Frame ${{frame.frameIndex+1}}/${{evidenceFrames.length}} · PTS ${{frame.sourcePTS}} · tick ${{frame.projectTick}} · ${{frame.timestampSeconds.toFixed(3)}} s`;
  if(frame.observationState==='missing'){{
    evidenceFlag.className='missing';
    evidenceFlag.textContent='MISSING · no face observation at this source frame'
  }}else{{
    evidenceFlag.className='unknown';
    evidenceFlag.textContent='UNKNOWN · observed tracker values are not a labeled neutral or expression state'
  }}
  for(const regionName of requiredEvidenceRegions){{
    evidenceRegionElements[regionName].textContent=confidenceText(frame.regions[regionName])
  }}
  refreshEvidenceControls()
}}
function stepEvidenceFrame(direction){{
  if(!evidenceReady||!media)return;
  const time=Number(media.currentTime),epsilon=1e-7;
  let target;
  if(direction<0){{
    target=0;
    for(let index=evidenceFrames.length-1;index>=0;index--){{
      if(evidenceFrames[index].timestampSeconds<time-epsilon){{target=index;break}}
    }}
  }}else{{
    target=evidenceFrames.length-1;
    for(let index=0;index<evidenceFrames.length;index++){{
      if(evidenceFrames[index].timestampSeconds>time+epsilon){{target=index;break}}
    }}
  }}
  const targetTime=evidenceFrames[target].timestampSeconds;
  media.pause();
  media.currentTime=targetTime;
  showEvidenceFrame(target);
  if(mixer&&animationAction){{
    animationAction.time=Math.min(Math.max(targetTime,0),animationDuration);
    mixer.update(0)
  }}
}}
async function loadPerformanceEvidence(){{
  if(!performanceEvidenceUrl||!media||mediaKind!=='video')return;
  evidencePanel.hidden=false;
  evidenceReadout.textContent='Loading source-frame evidence…';
  try{{
    const response=await fetch(validateEvidenceArtifactUrl(performanceEvidenceUrl),{{credentials:'same-origin',cache:'no-store',headers:{{Accept:'application/json'}}}});
    if(!response.ok)throw new Error(`Evidence request failed (${{response.status}})`);
    evidenceFrames=validatePerformanceEvidence(await response.json());
    evidenceReady=true;
    showEvidenceFrame(evidenceIndexAtOrBefore(media.currentTime));
    refreshEvidenceControls()
  }}catch(error){{
    evidenceError.hidden=false;
    evidenceError.textContent=`Evidence unavailable: ${{error.message}}`;
    evidenceReadout.textContent='Source-frame evidence unavailable';
    evidenceFlag.className='missing';
    evidenceFlag.textContent='MISSING · diagnostic artifact was not accepted'
  }}
}}
previousSourceFrame.addEventListener('click',()=>stepEvidenceFrame(-1));
nextSourceFrame.addEventListener('click',()=>stepEvidenceFrame(1));
if(media&&performanceEvidenceUrl&&mediaKind==='video'){{
  media.addEventListener('loadedmetadata',refreshEvidenceControls);
  media.addEventListener('seeked',()=>showEvidenceFrame(evidenceIndexAtOrBefore(media.currentTime)));
  media.addEventListener('timeupdate',()=>showEvidenceFrame(evidenceIndexAtOrBefore(media.currentTime)));
  void loadPerformanceEvidence()
}}
function cloneMaterials(material,wireOnly=false){{const list=Array.isArray(material)?material:[material];const cloned=list.map(source=>{{if(wireOnly)return new THREE.MeshBasicMaterial({{color:0xd8ff63,wireframe:true,transparent:true,opacity:.78}});const copy=source.clone();copy.wireframe=false;return copy}});return Array.isArray(material)?cloned:cloned[0]}}
function applyMode(){{if(!root)return;const mode=document.querySelector('#mode').value;root.traverse(node=>{{if(!node.isMesh)return;if(mode==='wire')node.material=node.userData.wireMaterial;else{{node.material=node.userData.surfaceMaterial;const mats=Array.isArray(node.material)?node.material:[node.material];mats.forEach(material=>material.wireframe=mode==='surface-wire')}}}})}}
function frameObject(object){{const box=new THREE.Box3().setFromObject(object),center=box.getCenter(new THREE.Vector3()),size=box.getSize(new THREE.Vector3()),radius=Math.max(size.x,size.y,size.z);homeTarget.copy(center);homePosition.set(center.x,center.y,center.z+radius*2.7);camera.near=Math.max(radius/500,.001);camera.far=Math.max(radius*20,2);camera.updateProjectionMatrix();camera.position.copy(homePosition);controls.minDistance=radius*1.2;controls.maxDistance=radius*8;controls.target.copy(homeTarget);controls.update();grid.position.set(center.x,box.min.y-.002,center.z)}}
new GLTFLoader().load(assetUrl,gltf=>{{root=gltf.scene;let vertices=0,triangles=0,meshes=0;root.traverse(node=>{{if(node.isMesh){{meshes++;const geometry=node.geometry;vertices+=geometry.getAttribute('position').count;triangles+=(geometry.index?geometry.index.count:geometry.getAttribute('position').count)/3;node.userData.surfaceMaterial=cloneMaterials(node.material);node.userData.wireMaterial=cloneMaterials(node.material,true)}}}});scene.add(root);if(gltf.animations.length){{const clip=gltf.animations[0];mixer=new THREE.AnimationMixer(root);animationAction=mixer.clipAction(clip);animationDuration=clip.duration;animationAction.setLoop(THREE.LoopOnce,1);animationAction.clampWhenFinished=true;animationAction.play();animationAction.paused=true;animationAction.time=0;mixer.update(0)}}frameObject(root);applyMode();document.querySelector('#metrics').textContent=`${{vertices.toLocaleString()}} render vertices · ${{Math.round(triangles).toLocaleString()}} triangles · ${{meshes}} primitive${{meshes===1?'':'s'}} · ${{gltf.animations.length?'animated':'static'}}`;status.textContent=mixer&&media?'Ready · media controls drive exact 3D time':'Ready · drag to orbit · scroll to zoom'}},undefined,error=>{{status.textContent=`Could not load 3D asset: ${{error.message}}`;status.style.color='#ff7d7d'}});
document.querySelector('#mode').addEventListener('change',applyMode);document.querySelector('#exposure').addEventListener('input',event=>renderer.toneMappingExposure=Number(event.target.value));document.querySelector('#reset').addEventListener('click',()=>{{camera.position.copy(homePosition);controls.target.copy(homeTarget);controls.update()}});
// Sample the paused action directly: AnimationMixer.setTime() zeroes action-local
// time, so a LoopOnce action that has reached the end cannot seek backward correctly.
function refreshPlaybackStatus(){{
  if(!mixer||!animationAction||!media)return;
  const time=media.currentTime.toFixed(2);
  if(!media.paused&&!media.ended)status.textContent=`Playing ${{time}} s · media-clock synchronized`;
  else if(media.ended)status.textContent=`Finished ${{time}} s · media-clock synchronized`;
  else if(media.currentTime<=.01)status.textContent='Ready · media controls drive exact 3D time';
  else status.textContent=`Paused ${{time}} s · media-clock synchronized`
}}
function resize(){{const width=stage.clientWidth,height=stage.clientHeight;renderer.setSize(width,height,false);camera.aspect=width/Math.max(height,1);camera.updateProjectionMatrix()}}new ResizeObserver(resize).observe(stage);resize();renderer.setAnimationLoop(()=>{{if(mixer&&animationAction&&media){{animationAction.time=Math.min(Math.max(media.currentTime,0),animationDuration);mixer.update(0);refreshPlaybackStatus()}}if(evidenceReady&&media)showEvidenceFrame(evidenceIndexAtOrBefore(media.currentTime));controls.update();renderer.render(scene,camera)}});
window.addEventListener('pagehide',()=>{{renderer.setAnimationLoop(null);controls.dispose();if(mixer)mixer.stopAllAction();if(root)root.traverse(node=>{{if(!node.isMesh)return;node.geometry?.dispose();const materials=[].concat(node.material||[],node.userData.surfaceMaterial||[],node.userData.wireMaterial||[]);for(const material of new Set(materials)){{material.map?.dispose();material.dispose?.()}}}});renderer.dispose()}});
</script></body></html>"""
