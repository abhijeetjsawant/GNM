"""Interactive local GLB viewer page."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path


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


def viewer_html(
    *,
    asset_url: str,
    title: str,
    media_url: str | None = None,
    media_type: str | None = None,
    vendor_base_url: str = f"/api/viewer/vendor/{VIEWER_THREE_VERSION}",
) -> str:
    """Return a self-contained shell that loads a job's allowlisted GLB."""

    safe_title = html.escape(title)
    encoded_url = json.dumps(asset_url)
    encoded_media_url = json.dumps(media_url)
    encoded_media_kind = json.dumps(
        "video" if media_type is not None and media_type.startswith("video/") else "audio"
    )
    encoded_three_url = json.dumps(f"{vendor_base_url}/three.module.js")
    encoded_addons_url = json.dumps(f"{vendor_base_url}/addons/")
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
    #stage{{min-width:0;min-height:0;position:relative}}canvas{{display:block;width:100%;height:100%}}
    #status{{position:absolute;left:18px;bottom:16px;padding:8px 11px;background:#080a0cdd;border:1px solid var(--line);border-radius:8px;color:var(--muted)}}
    .legend{{margin-top:24px;padding-top:16px;border-top:1px solid var(--line);font-size:12px}}
    @media(max-width:720px){{main{{grid-template-columns:1fr;grid-template-rows:auto 1fr}}aside{{padding:14px;border-right:0;border-bottom:1px solid var(--line)}}aside p,.legend{{display:none}}h1{{font-size:20px}}label{{display:inline-block;margin:6px 6px 4px 0}}select,input,button{{width:auto}}}}
  </style>
  <script type="importmap">{{"imports":{{"three":{encoded_three_url},"three/addons/":{encoded_addons_url}}}}}</script>
</head><body><main>
  <aside><h1>{safe_title}</h1><p>Orbit, zoom, and inspect the exact seam-correct GNM asset exported by this job.</p>
    <label for="mode">Display</label><select id="mode"><option value="surface">Surface / texture</option><option value="surface-wire">Surface + topology</option><option value="wire">Topology only</option></select>
    <label for="exposure">Exposure</label><input id="exposure" type="range" min="0.35" max="2.25" step="0.05" value="1.0">
    <button id="reset" type="button">Reset camera</button><div id="media-slot"></div>
    <div class="legend"><strong>GNM Head 3.0</strong><p id="metrics">Loading geometry…</p><p>Texture seams are split into duplicate render vertices that retain a mapping to the original GNM topology.</p></div>
  </aside>
  <section id="stage" aria-label="Interactive 3D GNM head"><div id="status" role="status" aria-live="polite">Loading 3D asset…</div></section>
</main>
<script type="module">
import * as THREE from 'three';
import {{OrbitControls}} from 'three/addons/controls/OrbitControls.js';
import {{GLTFLoader}} from 'three/addons/loaders/GLTFLoader.js';

const assetUrl={encoded_url},mediaUrl={encoded_media_url},mediaKind={encoded_media_kind},stage=document.querySelector('#stage'),status=document.querySelector('#status');
const scene=new THREE.Scene();scene.background=new THREE.Color(0x0b0e10);
const camera=new THREE.PerspectiveCamera(28,1,.005,20);
let renderer;try{{renderer=new THREE.WebGLRenderer({{antialias:true,alpha:false}})}}catch(error){{status.textContent='WebGL is unavailable. Download the GLB from the job result to inspect it in another viewer.';status.style.color='#ff7d7d';throw error}}renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1;renderer.domElement.setAttribute('role','img');renderer.domElement.setAttribute('aria-label','Interactive 3D GNM head');renderer.domElement.tabIndex=0;stage.prepend(renderer.domElement);
const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.dampingFactor=.075;controls.screenSpacePanning=true;controls.enablePan=false;controls.minPolarAngle=.2;controls.maxPolarAngle=Math.PI-.2;
scene.add(new THREE.HemisphereLight(0xeaf3ff,0x33291f,2.2));const key=new THREE.DirectionalLight(0xffffff,3.1);key.position.set(.8,.7,1.4);scene.add(key);const rim=new THREE.DirectionalLight(0xb7d8ff,1.8);rim.position.set(-1,.35,-1);scene.add(rim);
const grid=new THREE.GridHelper(1,20,0x4c555d,0x252b30);scene.add(grid);
let root=null,mixer=null,animationAction=null,animationDuration=0,homePosition=new THREE.Vector3(),homeTarget=new THREE.Vector3();
let media=null;if(mediaUrl){{const label=document.createElement('label');label.textContent=mediaKind==='video'?'Source performance':'Playback';media=document.createElement(mediaKind);media.controls=true;media.preload='metadata';if(mediaKind==='video')media.playsInline=true;media.src=mediaUrl;document.querySelector('#media-slot').append(label,media)}}
function cloneMaterials(material,wireOnly=false){{const list=Array.isArray(material)?material:[material];const cloned=list.map(source=>{{if(wireOnly)return new THREE.MeshBasicMaterial({{color:0xd8ff63,wireframe:true,transparent:true,opacity:.78}});const copy=source.clone();copy.wireframe=false;return copy}});return Array.isArray(material)?cloned:cloned[0]}}
function applyMode(){{if(!root)return;const mode=document.querySelector('#mode').value;root.traverse(node=>{{if(!node.isMesh)return;if(mode==='wire')node.material=node.userData.wireMaterial;else{{node.material=node.userData.surfaceMaterial;const mats=Array.isArray(node.material)?node.material:[node.material];mats.forEach(material=>material.wireframe=mode==='surface-wire')}}}})}}
function frameObject(object){{const box=new THREE.Box3().setFromObject(object),center=box.getCenter(new THREE.Vector3()),size=box.getSize(new THREE.Vector3()),radius=Math.max(size.x,size.y,size.z);homeTarget.copy(center);homePosition.set(center.x,center.y,center.z+radius*2.7);camera.near=Math.max(radius/500,.001);camera.far=Math.max(radius*20,2);camera.updateProjectionMatrix();camera.position.copy(homePosition);controls.minDistance=radius*1.2;controls.maxDistance=radius*8;controls.target.copy(homeTarget);controls.update();grid.position.set(center.x,box.min.y-.002,center.z)}}
new GLTFLoader().load(assetUrl,gltf=>{{root=gltf.scene;let vertices=0,triangles=0,meshes=0;root.traverse(node=>{{if(node.isMesh){{meshes++;const geometry=node.geometry;vertices+=geometry.getAttribute('position').count;triangles+=(geometry.index?geometry.index.count:geometry.getAttribute('position').count)/3;node.userData.surfaceMaterial=cloneMaterials(node.material);node.userData.wireMaterial=cloneMaterials(node.material,true)}}}});scene.add(root);if(gltf.animations.length){{const clip=gltf.animations[0];mixer=new THREE.AnimationMixer(root);animationAction=mixer.clipAction(clip);animationDuration=clip.duration;animationAction.setLoop(THREE.LoopOnce,1);animationAction.clampWhenFinished=true;animationAction.play();animationAction.paused=true;animationAction.time=0;mixer.update(0)}}frameObject(root);applyMode();document.querySelector('#metrics').textContent=`${{vertices.toLocaleString()}} render vertices · ${{Math.round(triangles).toLocaleString()}} triangles · ${{meshes}} primitive${{meshes===1?'':'s'}} · ${{gltf.animations.length?'animated':'static'}}`;status.textContent=mixer&&media?'Ready · media controls drive exact 3D time':'Ready · drag to orbit · scroll to zoom'}},undefined,error=>{{status.textContent=`Could not load 3D asset: ${{error.message}}`;status.style.color='#ff7d7d'}});
document.querySelector('#mode').addEventListener('change',applyMode);document.querySelector('#exposure').addEventListener('input',event=>renderer.toneMappingExposure=Number(event.target.value));document.querySelector('#reset').addEventListener('click',()=>{{camera.position.copy(homePosition);controls.target.copy(homeTarget);controls.update()}});
// Sample the paused action directly: AnimationMixer.setTime() zeroes action-local
// time, so a LoopOnce action that has reached the end cannot seek backward correctly.
function resize(){{const width=stage.clientWidth,height=stage.clientHeight;renderer.setSize(width,height,false);camera.aspect=width/Math.max(height,1);camera.updateProjectionMatrix()}}new ResizeObserver(resize).observe(stage);resize();renderer.setAnimationLoop(()=>{{if(mixer&&animationAction&&media){{animationAction.time=Math.min(Math.max(media.currentTime,0),animationDuration);mixer.update(0);if(!media.paused)status.textContent=`Playing ${{media.currentTime.toFixed(2)}} s · media-clock synchronized`}}controls.update();renderer.render(scene,camera)}});
window.addEventListener('pagehide',()=>{{renderer.setAnimationLoop(null);controls.dispose();if(mixer)mixer.stopAllAction();if(root)root.traverse(node=>{{if(!node.isMesh)return;node.geometry?.dispose();const materials=[].concat(node.material||[],node.userData.surfaceMaterial||[],node.userData.wireMaterial||[]);for(const material of new Set(materials)){{material.map?.dispose();material.dispose?.()}}}});renderer.dispose()}});
</script></body></html>"""
