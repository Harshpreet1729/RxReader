// Progressive enhancement: server-rendered forms also work without JavaScript.
const statusBox = document.querySelector('#job-status');
if (statusBox?.dataset.busy === 'true') {
  const poll = async () => {
    try {
      const response = await fetch(statusBox.dataset.statusUrl, {cache: 'no-store'});
      if (!response.ok) { document.querySelector('#stage').textContent = 'This prescription expired. Upload it again.'; return; }
      const job = await response.json();
      document.querySelector('#stage').textContent = job.stage;
      const bar = document.querySelector('#reading-progress');
      if (job.total) { bar.max = job.total; bar.value = job.completed; }
      document.querySelector('#count').textContent = job.total ? `${job.completed} / ${job.total} regions` : '';
      if (['ready','failed'].includes(job.state)) { location.reload(); return; }
    } catch { document.querySelector('#stage').textContent = 'Connection interrupted. Trying again…'; }
    setTimeout(poll, 1500);
  };
  setTimeout(poll, 800);
}
const image = document.querySelector('#prescription');
const overlay = document.querySelector('#overlay');
const regions = JSON.parse(document.querySelector('#regions-data')?.textContent || '[]');
const ns = 'http://www.w3.org/2000/svg';
function rectangle(box, color, fill = 'transparent') {
  const rect = document.createElementNS(ns,'rect');
  const [x1,y1,x2,y2] = box;
  Object.entries({x:x1,y:y1,width:x2-x1,height:y2-y1,stroke:color,fill,'stroke-width':3,'vector-effect':'non-scaling-stroke'}).forEach(([k,v])=>rect.setAttribute(k,v));
  return rect;
}
function showRegion(id) {
  overlay.replaceChildren();
  const region = regions.find(r=>r.id===id);
  if (region) { overlay.append(rectangle(region.box,'#087f8c')); image.scrollIntoView({behavior:'smooth',block:'center'}); }
}
document.querySelectorAll('[data-highlight]').forEach(button=>button.addEventListener('click',()=>showRegion(button.dataset.highlight)));
// Changing evidence invalidates its previous review checkbox.
document.querySelectorAll('#review-form textarea, #review-form select').forEach(field=>field.addEventListener('input',()=>{
  field.closest('.region').querySelector('input[type=checkbox]').checked=false;
}));
const canvas=document.querySelector('#page-image');
const draw=document.querySelector('#draw');
let drawing=false, start=null;
function point(event) {
  const r=image.getBoundingClientRect();
  return [Math.max(0,Math.min(1,(event.clientX-r.left)/r.width)),Math.max(0,Math.min(1,(event.clientY-r.top)/r.height))];
}
draw?.addEventListener('click',()=>{drawing=!drawing;canvas.classList.toggle('drawing',drawing);draw.textContent=drawing?'Drag over the text…':'Draw region';});
canvas?.addEventListener('pointerdown',event=>{if(!drawing)return;start=point(event);canvas.setPointerCapture(event.pointerId);event.preventDefault();});
canvas?.addEventListener('pointermove',event=>{
  if(!start)return;
  const end=point(event), box=[Math.min(start[0],end[0]),Math.min(start[1],end[1]),Math.max(start[0],end[0]),Math.max(start[1],end[1])];
  const vb=overlay.viewBox.baseVal;
  overlay.replaceChildren(rectangle([box[0]*vb.width,box[1]*vb.height,box[2]*vb.width,box[3]*vb.height],'#087f8c','#087f8c22'));
  ['x1','y1','x2','y2'].forEach((name,i)=>document.querySelector(`[name="${name}"]`).value=box[i].toFixed(6));
});
canvas?.addEventListener('pointerup',()=>{if(!start)return;start=null;drawing=false;canvas.classList.remove('drawing');draw.textContent='Draw region';});
canvas?.addEventListener('pointercancel',()=>{start=null;});
