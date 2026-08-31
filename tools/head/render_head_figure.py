"""Head-orientation overlay, rendered honestly.

Corrections applied after verification found three defects in the first attempt:
  * the gizmo was drawn at the TEMPLATE ORIGIN, which sits at the neck joint, not on the
    skull. It is now drawn at the fitted world position of the `Head` landmark.
  * frames were greedy-picked to MAXIMISE the fit-vs-locked angle. They are now evenly
    spaced, and the chosen indices are printed on the figure.
  * only camera A001 was shown, and in 7 of 12 panels the difference lay along that
    camera's viewing ray, so the drawn arrows separated by <= 7 px while the caption
    claimed 13-35 deg. ALL FOUR cameras are now shown for the same instant, so the
    reader sees directly where the difference is visible and where projection hides it.
"""
import json, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, '/Users/abhi_macbook/Projects/apps/AutoAnim/tools/head')
sys.path.insert(0, '/Users/abhi_macbook/Projects/apps/AutoAnim/src')
import os; os.chdir('/Users/abhi_macbook/Projects/apps/AutoAnim')
from autoanim_gnm.commercial_multiview import JOINT_INDEX, load_camera_rig, _thorax_frames
from head_gate import frame_from
from mamma_head_bar import geodesic_deg

OUT = '/private/tmp/claude-501/-Users-abhi-macbook-Projects-apps-AutoAnim/da9cede3-fc40-4418-88ca-da28b459c7fe/scratchpad/fig'
CAMS = ('A001','B001','C001','D001')
NAMES = ["Head","HeadEnd","Jaw","LeftEye","RightEye"]
FRAMES = [30, 75, 120]           # evenly spaced, chosen before any angle was computed
AXIS_M = 0.14
COL_FIT = {'up':(255,214,10),'right':(80,200,255),'fwd':(255,90,120)}
COL_LOCK = (150,150,150)

rig = {c.name: c for c in load_camera_rig('artifacts/soma77-full/camera-rig.json')}
cams = {n: rig[n].scaled(1280,720) for n in CAMS}
# The SHIPPED solve, not the prototype's. This read `head-solve.npz` -- a different
# estimator from the one the pipeline delivers -- so the gizmos drawn here were not
# the head anyone receives. Same trap as the gate's old default; see §6j.
blob = np.load('artifacts/head-lane/head-solve-shipped.npz')

def font(sz):
    for p in ('/System/Library/Fonts/Supplemental/Arial Bold.ttf','/System/Library/Fonts/Helvetica.ttc'):
        try: return ImageFont.truetype(p, sz)
        except Exception: pass
    return ImageFont.load_default()

state = {}
for s in (0,1):
    R = blob[f'subject_{s:02d}_head_world']; P = blob[f'subject_{s:02d}_head_position_m']
    T = blob[f'subject_{s:02d}_template_m']
    sm = np.load(f'artifacts/commercial-multiview-soma77/subject-{s:02d}.body-track.npz')['triangulated_world_positions_z_up_m']
    # The PIPELINE's torso frame, imported rather than rebuilt. This used to call
    # frame_from on the raw smoothed landmarks, which is a second definition of one
    # quantity -- and two copies is how the figure and the gate come to print different
    # "take medians" for the same head.
    th = _thorax_frames(sm)
    up_l = T[NAMES.index('HeadEnd')] - T[NAMES.index('Head')]; up_l /= np.linalg.norm(up_l)
    r_l  = T[NAMES.index('LeftEye')] - T[NAMES.index('RightEye')]
    r_l  = r_l - up_l*(r_l@up_l); r_l /= np.linalg.norm(r_l)
    f_l  = np.cross(up_l, r_l)
    # skull-interior origin: the Head landmark, not the template centroid
    head_world = P + np.einsum('nij,j->ni', R, T[NAMES.index('Head')])
    # The head's local frame and the thorax frame differ by an arbitrary CONSTANT (here
    # ~104 deg / ~77 deg). That constant is gauge, not motion, and the gate removes it
    # before scoring. So the locked head is drawn as thorax(t) @ M, where M is the take's
    # mean relative pose -- i.e. "the head frozen at its average pose on this body". The
    # two gizmos then coincide exactly when the head sits at its take mean, and the angle
    # printed is the mean-removed deviation, which is the gate's P2 quantity.
    rel = np.einsum('nji,njk->nik', th, R)
    u_, _, vt_ = np.linalg.svd(rel.mean(axis=0)); M = u_ @ vt_
    if np.linalg.det(M) < 0: u_[:, -1] *= -1; M = u_ @ vt_
    locked_world = np.einsum('nij,jk->nik', th, M)
    state[s] = dict(R=R, th=th, locked=locked_world, head=head_world,
                    local=dict(up=up_l, right=r_l, fwd=f_l),
                    ang=geodesic_deg(np.einsum('nij,kj->nik', rel, M),
                                     np.broadcast_to(np.eye(3), R.shape)))

def draw_gizmo(dr, cam, origin, Rw, local, scale, ox, oy, dashed, width):
    o_uv, o_d = cam.project(origin[None])
    o_uv = np.asarray(o_uv).reshape(-1)[:2]
    if np.asarray(o_d).reshape(-1)[0] <= 0: return None
    tips = {}
    for name, d in local.items():
        tip = origin + (Rw @ d) * AXIS_M
        t_uv, t_d = cam.project(tip[None])
        t_uv = np.asarray(t_uv).reshape(-1)[:2]
        if np.asarray(t_d).reshape(-1)[0] <= 0: continue
        a = ((o_uv[0]-ox)*scale, (o_uv[1]-oy)*scale); b = ((t_uv[0]-ox)*scale, (t_uv[1]-oy)*scale)
        col = COL_LOCK if dashed else COL_FIT[name]
        if dashed:
            n = 14; 
            for k in range(n):
                if k % 2: continue
                p0=(a[0]+(b[0]-a[0])*k/n, a[1]+(b[1]-a[1])*k/n)
                p1=(a[0]+(b[0]-a[0])*(k+1)/n, a[1]+(b[1]-a[1])*(k+1)/n)
                dr.line([p0,p1], fill=col, width=width)
        else:
            dr.line([a,b], fill=col, width=width)
            dr.ellipse([b[0]-4,b[1]-4,b[0]+4,b[1]+4], fill=col)
        tips[name]=b
    dr.ellipse([ (o_uv[0]-ox)*scale-3, (o_uv[1]-oy)*scale-3,
                 (o_uv[0]-ox)*scale+3, (o_uv[1]-oy)*scale+3 ], fill=(255,255,255))
    return tips

manifest=[]
for frame in FRAMES:
    PAN, GAP, HEADER = 300, 8, 62
    W = PAN*4 + GAP*3; H = HEADER + PAN*2 + GAP + 52
    sheet = Image.new('RGB',(W,H),(14,16,20)); sd = ImageDraw.Draw(sheet)
    sd.text((14,12), f"Solve frame {frame} of 150  ·  plate {60+frame:06d}  ·  all four cameras, same instant",
            font=font(19), fill=(238,240,245))
    sd.text((14,38), "solid = the fitted head   ·   grey dashed = the shipped head: CARRIED BY THE TORSO, holding this take's average head pose   ·   yellow up, blue right, red forward",
            font=font(13), fill=(150,158,170))
    seps=[]
    for ci,cn in enumerate(CAMS):
        img = Image.open(f'artifacts/commercial-multiview-soma77/work/frames/{cn}/{60+frame:06d}.jpg').convert('RGB')
        cam = cams[cn]
        for si,s in enumerate((0,1)):
            st = state[s]; origin = st['head'][frame]
            uv,d = cam.project(origin[None])
            uv = np.asarray(uv).reshape(-1)[:2]
            if np.asarray(d).reshape(-1)[0] <= 0:
                continue
            cx,cy = uv; box=150
            ox,oy = cx-box/2, cy-box/2
            crop = img.crop((int(ox),int(oy),int(ox+box),int(oy+box))).resize((PAN,PAN), Image.LANCZOS)
            scale = PAN/box
            dr = ImageDraw.Draw(crop)
            lock = draw_gizmo(dr, cam, origin, st['locked'][frame], st['local'], scale, ox, oy, True, 3)
            fit  = draw_gizmo(dr, cam, origin, st['R'][frame],  st['local'], scale, ox, oy, False, 3)
            if lock and fit:
                seps += [np.hypot(fit[k][0]-lock[k][0], fit[k][1]-lock[k][1])/scale for k in fit if k in lock]
            dr.rectangle([0,0,PAN-1,PAN-1], outline=(60,66,76))
            dr.text((7,6), f"{cn}  ·  performer {s}", font=font(14), fill=(238,240,245))
            sheet.paste(crop, (ci*(PAN+GAP), HEADER + si*(PAN+GAP)))
    med = float(np.median(seps)) if seps else float('nan')
    a0,a1 = state[0]['ang'][frame], state[1]['ang'][frame]
    sd.text((14, H-44),
            f"Head turn away from its own take-average, this frame: {a0:.1f}° (performer 0), {a1:.1f}° (performer 1). "
            f"Take medians {np.median(state[0]['ang']):.1f}° and {np.median(state[1]['ang']):.1f}°.",
            font=font(12), fill=(150,158,170))
    sd.text((14, H-26),
            f"Median projected separation between the two gizmos across these four views: {med:.0f} px at source scale. Where it looks small, projection is hiding "
            f"the angle along that camera's ray — a picture cannot show depth, which is why the gate is scored in 3-D and not from images like this one.",
            font=font(12), fill=(150,158,170))
    path=f'{OUT}/heads_f{frame:03d}.jpg'
    sheet.save(path, quality=86, optimize=True)
    manifest.append(dict(frame=frame, plate=60+frame, path=path,
                         angle_s0=round(float(a0),2), angle_s1=round(float(a1),2),
                         median_arrow_separation_px=round(med,1),
                         bytes=os.path.getsize(path), size=sheet.size))
    print(f"frame {frame}: {a0:.1f}/{a1:.1f} deg, median sep {med:.1f} px, {os.path.getsize(path)//1024} KB")

json.dump(dict(frames_chosen="evenly spaced (30, 75, 120) BEFORE any angle was computed; no selection on the displayed quantity",
               gizmo_origin="the fitted world position of the Head landmark (skull interior), not the template centroid",
               cameras="all four, same instant, so the reader sees where projection hides the difference",
               figures=manifest), open(f'{OUT}/manifest.json','w'), indent=2)
print('take medians:', [round(float(np.median(state[s]['ang'])),2) for s in (0,1)])
