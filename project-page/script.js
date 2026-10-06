const root = '../figures/video_demo/gifs/';
const cases = {rca:[476,785,871,969,289,457,713,806,837],lca:[290,457,526,560,154,444,663,858,959]};
let artery = 'rca';
let moving = !matchMedia('(prefers-reduced-motion: reduce)').matches;
let caseIndex = 0;
function sourceImage(image, src, still, alt) { image.dataset.motion=src; image.dataset.still=still; image.alt=alt; image.src=moving?src:still; }
function showCase() {
 const n=cases[artery][caseIndex], name=artery.toUpperCase();
 document.querySelector('#case-description').textContent = `${name==='RCA'?'Right':'Left'} coronary artery · Example ${caseIndex+1}`;
 const inputs=document.querySelector('#case-inputs');inputs.replaceChildren();
 for(let v=0;v<2;v++){const img=new Image();img.src=`${root}input_views/${artery}/${n}_view_${v}.png`;img.alt=`${name} artery, segmented input view ${v+1}`;inputs.append(img);}
 sourceImage(document.querySelector('#label-gif'),`${root}ours_radius_colored/gt/gt_${artery}_${n}_radius.gif`,`assets/gt_${artery}_${n}.png`,`${name} artery: 3D artery label coloured by radius`);
 sourceImage(document.querySelector('#prediction-gif'),`${root}ours_radius_colored/ours_k2/ours_${artery}_${n}_radius.gif`,`assets/ours_${artery}_${n}.png`,`${name} artery: our two-view reconstruction coloured by radius`);
 document.querySelector('#colorbar').src=`${root}ours_radius_colored/colorbars/${artery}_${n}_colorbar.png`;
}
function setArtery(value){artery=value;caseIndex=0;document.querySelectorAll('[data-artery]').forEach(b=>b.setAttribute('aria-pressed',b.dataset.artery===artery));showCase();}
function updateMotion(){document.querySelectorAll('.animated').forEach(img=>img.src=moving?img.dataset.motion:img.dataset.still);}
setArtery(artery);updateMotion();
document.querySelector('#previous-case').addEventListener('click',()=>{caseIndex=(caseIndex-1+cases[artery].length)%cases[artery].length;showCase();});
document.querySelector('#next-case').addEventListener('click',()=>{caseIndex=(caseIndex+1)%cases[artery].length;showCase();});
document.querySelectorAll('[data-artery]').forEach(b=>b.addEventListener('click',()=>setArtery(b.dataset.artery)));
const dialog=document.querySelector('#lightbox');
document.querySelectorAll('.zoom').forEach(b=>b.addEventListener('click',()=>{const image=b.querySelector('img');dialog.querySelector('img').src=image.src;dialog.querySelector('img').alt=image.alt;dialog.showModal();}));
document.querySelector('#close-lightbox').addEventListener('click',()=>dialog.close());
dialog.addEventListener('click',e=>{if(e.target===dialog)dialog.close();});

function setupSectionRotation(sectionId, buttonId){
 let playing=!matchMedia('(prefers-reduced-motion: reduce)').matches;
 const button=document.getElementById(buttonId);
 function update(){document.querySelectorAll(`#${sectionId} .transfer-animation`).forEach(img=>img.src=playing?img.dataset.motion:img.dataset.still);if(button){button.textContent=playing?'Pause rotation':'Play rotation';button.setAttribute('aria-pressed',String(playing));}}
 if(button)button.addEventListener('click',()=>{playing=!playing;update();});update();
}
setupSectionRotation('transfer','transfer-motion');
setupSectionRotation('real-xray','real-motion');


const sectionLinks=[...document.querySelectorAll('.section-sidebar a')];
const sectionTargets=sectionLinks.map(link=>document.querySelector(link.getAttribute('href')));
let sectionTick=false;
function highlightSection(){
 let current=sectionTargets[0];
 for(const section of sectionTargets)if(section.getBoundingClientRect().top<=150)current=section;
 if(innerHeight+scrollY>=document.documentElement.scrollHeight-4)current=sectionTargets[sectionTargets.length-1];
 sectionLinks.forEach(link=>{if(link.hash==='#'+current.id)link.setAttribute('aria-current','location');else link.removeAttribute('aria-current');});
 sectionTick=false;
}
addEventListener('scroll',()=>{if(!sectionTick){sectionTick=true;requestAnimationFrame(highlightSection);}},{passive:true});
addEventListener('resize',highlightSection);highlightSection();
