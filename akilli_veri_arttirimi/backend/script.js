document.addEventListener('DOMContentLoaded', () => {
    const API = window.location.protocol === 'file:' ? 'http://127.0.0.1:8000' : '';
    const $ = id => document.getElementById(id);
    let currentFile = null, f1Chart = null, distilledFile = null;
    
    // Set absolute download links
    $('dl-distilled').href = API + '/api/download_distilled';
    document.querySelector('#dl-card .btn-download').href = API + '/api/download_generated';

    function getDesktopApi() {
        return window.pywebview && window.pywebview.api ? window.pywebview.api : null;
    }

    async function saveCsvViaDesktop(kind) {
        const method = kind === 'distilled' ? 'save_distilled_csv' : 'save_generated_csv';
        const desktopApi = getDesktopApi();
        if (!desktopApi || !desktopApi[method]) {
            log('Masaüstü kaydetme servisi hazır değil. Uygulamayı main.py ile açın.', 'err');
            return false;
        }

        const result = await desktopApi[method]();
        if (result && result.ok) {
            log('CSV kaydedildi: ' + result.path, 'ok');
        } else if (result && !result.cancelled) {
            log(result.message || 'CSV kaydedilemedi.', 'err');
        }
        return true;
    }

    $('dl-distilled').addEventListener('click', async e => {
        if (getDesktopApi()) {
            e.preventDefault();
            await saveCsvViaDesktop('distilled');
        }
    });
    document.querySelector('#dl-card .btn-download').addEventListener('click', async e => {
        if (getDesktopApi()) {
            e.preventDefault();
            await saveCsvViaDesktop('generated');
        }
    });

    // === NAV ===
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', () => {
            document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
            document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
            item.classList.add('active');
            $('panel-' + item.dataset.panel).classList.add('active');
            $('panel-title').textContent = item.querySelector('span').textContent;
        });
    });

    // === CLOCK ===
    setInterval(() => { $('clock').textContent = new Date().toLocaleTimeString('tr-TR',{hour12:false}); }, 1000);

    // === LOG ===
    function log(msg, type='') {
        const d = document.createElement('div');
        d.className = 'clog ' + type;
        const t = new Date().toLocaleTimeString('tr-TR',{hour12:false});
        d.innerHTML = `<span style="opacity:.4">[${t}]</span> ${msg}`;
        $('console-out').appendChild(d);
        $('console-out').scrollTop = 99999;
    }

    // === STATUS ===
    async function checkStatus() {
        try {
            const r = await fetch(API+'/api/system_status');
            const d = await r.json();
            if(d.model_loaded){
                $('sys-indicator').innerHTML='<span class="blink-dot green"></span>';
                $('sys-label').textContent='GODMODE Aktif';$('sys-label').style.color='var(--green)';
                log('RCGAN GODMODE yüklendi: '+d.model_name,'ok');
            } else { $('sys-label').textContent='Fallback';$('sys-label').style.color='var(--amber)'; }
            $('sys-seed').textContent=d.seed_rows.toLocaleString();
            log('Seed: '+d.seed_rows.toLocaleString()+' yörünge');
        } catch { $('sys-label').textContent='Bağlantı Yok';$('sys-label').style.color='var(--red)'; log('Sunucuya bağlanılamadı!','err'); }
    }
    checkStatus();

    // === CHART ===
    (function(){
        const ctx=$('f1Chart').getContext('2d');
        Chart.defaults.color='#5a6578';Chart.defaults.font.family='Inter';
        f1Chart=new Chart(ctx,{type:'bar',data:{labels:['Baseline','Augmented'],datasets:[{data:[0,0],backgroundColor:['rgba(99,102,241,.45)','rgba(16,185,129,.6)'],borderColor:['#6366f1','#10b981'],borderWidth:2,borderRadius:6,barPercentage:.5}]},options:{responsive:true,maintainAspectRatio:false,scales:{y:{beginAtZero:true,max:1,grid:{color:'rgba(255,255,255,.03)'}},x:{grid:{display:false}}},plugins:{legend:{display:false}}}});
    })();

    // === SLIDER ===
    $('n-samples').addEventListener('input', e => { $('slider-val').textContent = e.target.value; });

    // === UPLOAD ===
    const dz=$('drop-zone');
    dz.addEventListener('click',()=>$('file-input').click());
    dz.addEventListener('dragover',e=>{e.preventDefault();dz.classList.add('over')});
    dz.addEventListener('dragleave',()=>dz.classList.remove('over'));
    dz.addEventListener('drop',e=>{e.preventDefault();dz.classList.remove('over');if(e.dataTransfer.files.length)loadFile(e.dataTransfer.files[0])});
    $('file-input').addEventListener('change',e=>{if(e.target.files.length)loadFile(e.target.files[0])});
    $('btn-remove').addEventListener('click',()=>{
        currentFile=null;distilledFile=null;$('file-input').value='';
        dz.classList.remove('hidden');$('file-pill').classList.add('hidden');
        $('btn-generate').disabled=true;$('btn-distill').disabled=true;
        log('Dosya kaldırıldı.');
    });

    function loadFile(f) {
        if(!f.name.toLowerCase().endsWith('.csv')){
            log('Sadece CSV dosyası yüklenebilir. Seçilen dosya: '+f.name, 'err');
            return;
        }

        currentFile=f;
        $('fname').textContent=f.name;$('fsize').textContent=(f.size/1024).toFixed(0)+' KB';
        dz.classList.add('hidden');$('file-pill').classList.remove('hidden');
        $('btn-distill').disabled=false;
        $('btn-generate').disabled=false;  // generate also works without distill
        log('Dosya: '+f.name+' ('+(f.size/1024).toFixed(0)+' KB)','ok');
    }

    // === DISTILL ===
    $('btn-distill').addEventListener('click', async () => {
        if(!currentFile) return;
        const btn=$('btn-distill');
        btn.disabled=true; btn.innerHTML='<i class="fa-solid fa-spinner fa-spin"></i> Damıtılıyor...';
        $('distill-progress').classList.remove('hidden');
        log('Bilgi damıtma başlatıldı...','ok');

        let p=0;
        const iv=setInterval(()=>{p+=Math.random()*15;if(p>90)p=90;$('distill-thumb').style.width=p+'%'},300);

        try {
            const fd=new FormData(); fd.append('file',currentFile);
            const r=await fetch(API+'/api/distill',{method:'POST',body:fd});
            if(!r.ok) throw new Error('Damıtma hatası: '+r.status);
            const d=await r.json();
            clearInterval(iv);
            $('distill-thumb').style.width='100%';
            $('distill-text').textContent='Damıtma tamamlandı!';
            btn.innerHTML='<i class="fa-solid fa-check"></i> Damıtma Tamamlandı';

            // Build report
            const rpt=$('distill-report'); rpt.innerHTML='';
            const rep=d.report;

            // Original info
            addDistillRow(rpt,'Orijinal Veri', rep.original_rows.toLocaleString()+' satır, '+rep.original_cols+' sütun','');

            // Steps
            rep.steps.forEach(s=>{
                let val='', cls='';
                if(s.removed!==undefined){val='-'+s.removed;cls='removed';}
                else if(s.fixed!==undefined){val='~'+s.fixed+' düzeltildi';cls='fixed';}
                else {val=s.detail;cls='';}
                addDistillRow(rpt,s.name,val,cls);
                log('🔧 '+s.name+': '+s.detail);
            });

            // Summary
            const sum=document.createElement('div');
            sum.className='distill-summary';
            sum.textContent='✅ Temiz veri: '+rep.clean_rows.toLocaleString()+' satır ('+rep.reduction_pct+'% azaltma)';
            rpt.appendChild(sum);

            $('dl-distilled').classList.remove('hidden');
            log('Damıtma tamamlandı: '+rep.original_rows+' → '+rep.clean_rows+' satır ('+rep.reduction_pct+'% azaltma)','ok');

            // Store distilled file info for generate
            distilledFile = true;
        } catch(e) {
            clearInterval(iv); log(e.message,'err');
            btn.disabled=false; btn.innerHTML='<i class="fa-solid fa-broom"></i> Veriyi Damıt';
        }
    });

    function addDistillRow(container, name, val, cls) {
        const r=document.createElement('div'); r.className='distill-row';
        r.innerHTML=`<span class="d-name">${name}</span><span class="d-val ${cls}">${val}</span>`;
        container.appendChild(r);
    }

    // === GENERATE ===
    $('btn-generate').addEventListener('click', async () => {
        if(!currentFile) return;
        const btn=$('btn-generate');
        const nSamples=parseInt($('n-samples').value);
        btn.disabled=true; btn.innerHTML='<i class="fa-solid fa-spinner fa-spin"></i> İşleniyor...';
        $('gen-progress').classList.remove('hidden');
        log('Sentez başlatıldı ('+nSamples+' örnek)...','ok');

        let p=0;
        const iv=setInterval(()=>{p+=Math.random()*10;if(p>92)p=92;$('gen-thumb').style.width=p+'%'},400);

        try {
            const fd=new FormData();
            fd.append('file',currentFile);
            fd.append('n_samples', nSamples.toString());
            const r=await fetch(API+'/api/evaluate_pipeline',{method:'POST',body:fd});
            if(!r.ok){ const err=await r.json(); throw new Error(err.detail||'Hata: '+r.status); }
            const d=await r.json();
            clearInterval(iv);
            $('gen-thumb').style.width='100%';$('gen-text').textContent='Tamamlandı!';
            btn.innerHTML='<i class="fa-solid fa-check"></i> Tamamlandı';
            showResults(d);
        } catch(e) {
            clearInterval(iv); log(e.message,'err');
            btn.disabled=false; btn.innerHTML='<i class="fa-solid fa-play"></i> Üretimi Başlat';
        }
    });

    function showResults(d) {
        $('kpi-seed').textContent=d.seed_count.toLocaleString();
        $('kpi-gen').textContent=d.gen_count.toLocaleString();
        $('kpi-factor').textContent='×'+d.multiplication_factor;
        $('kpi-cov').textContent='%'+d.generative_coverage;
        $('kpi-f1').textContent=d.augmented_f1.toFixed(4);
        const imp=$('kpi-imp');
        imp.textContent=(d.improvement>0?'+':'')+d.improvement.toFixed(1)+'%';
        imp.style.color=d.improvement>0?'var(--green)':'var(--red)';

        const methodNames={rcgan:'RCGAN GODMODE',ctgan:'CTGAN (On-the-fly)',smote:'SMOTE+Gaussian'};
        f1Chart.data.labels=['Baseline',methodNames[d.method]||'Augmented'];
        f1Chart.data.datasets[0].data=[d.seed_f1,d.augmented_f1];
        f1Chart.update();

        // Details
        const dl=$('detail-list'); dl.innerHTML='';
        if(d.analysis_note){
            const r=document.createElement('div');r.className='detail-row';
            r.innerHTML=`<span class="label" style="color:var(--amber);font-weight:700">Analiz Notu</span><span class="val">${d.analysis_note}</span>`;
            dl.appendChild(r);
        }
        if(d.dataset_info){
            const di=d.dataset_info;
            [['Satır (temiz)',d.seed_count.toLocaleString()],['Özellik',di.features],['Sınıf',di.classes],
             ['Label','"'+di.label_col+'"'],['Format',di.is_waymo?'Waymo':'Genel'],
             ['Yöntem',methodNames[d.method]||d.method],
             ['Seed F1',d.seed_f1.toFixed(4)],['Aug F1',d.augmented_f1.toFixed(4)]
            ].forEach(([l,v])=>{
                const r=document.createElement('div');r.className='detail-row';
                r.innerHTML=`<span class="label">${l}</span><span class="val">${v}</span>`;
                dl.appendChild(r);
            });
        }

        if(d.quality_report){
            const qr=d.quality_report;
            const hdrQ=document.createElement('div');hdrQ.className='detail-row';
            hdrQ.innerHTML='<span class="label" style="color:var(--cyan);font-weight:700">── Kalite Skoru ──</span><span class="val">'+qr.overall_score+'/100 ('+qr.grade+')</span>';
            dl.appendChild(hdrQ);
            [['Yönlendirme',qr.routing_explanation],
             ['Fidelity Skoru',qr.components?.fidelity?.score!=null?qr.components.fidelity.score+'/100':'Uygulanamaz'],
             ['Utility Skoru',qr.components?.utility?.score!=null?qr.components.utility.score+'/100':'Uygulanamaz'],
             ['Dağılım Skoru',qr.components?.distribution?.score!=null?qr.components.distribution.score+'/100':'Uygulanamaz'],
             ['Fizik Skoru',qr.components?.physical?.score!=null?qr.components.physical.score+'/100':'Uygulanamaz']
            ].forEach(([l,v])=>{
                const r=document.createElement('div');r.className='detail-row';
                r.innerHTML=`<span class="label">${l}</span><span class="val">${v}</span>`;
                dl.appendChild(r);
            });
            if(qr.distribution_shift && qr.distribution_shift.warnings && qr.distribution_shift.warnings.length){
                qr.distribution_shift.warnings.slice(0,4).forEach(w=>{
                    const r=document.createElement('div');r.className='detail-row';
                    r.innerHTML=`<span class="label" style="color:var(--amber)">Dağılım Uyarısı</span><span class="val">${w.detail}</span>`;
                    dl.appendChild(r);
                });
            }
            if(qr.scientific_basis){
                const r=document.createElement('div');r.className='detail-row';
                r.innerHTML='<span class="label">Bilimsel Temel</span><span class="val">Fidelity + Utility + Distribution Shift + Fiziksel Tutarlılık</span>';
                dl.appendChild(r);
            }
        }

        // Distillation report in details
        if(d.distillation && d.distillation.steps){
            const hdr=document.createElement('div');hdr.className='detail-row';
            hdr.innerHTML='<span class="label" style="color:var(--cyan);font-weight:700">── Damıtma ──</span><span></span>';
            dl.appendChild(hdr);
            d.distillation.steps.forEach(s=>{
                const r=document.createElement('div');r.className='detail-row';
                r.innerHTML=`<span class="label">${s.name}</span><span class="val">${s.detail}</span>`;
                dl.appendChild(r);
            });
        }
        
        // Fidelity & Utility (Akademik Metrikler)
        if (d.fidelity && d.utility) {
            // Fidelity
            const hdrF=document.createElement('div');hdrF.className='detail-row';
            hdrF.innerHTML='<span class="label" style="color:var(--green);font-weight:700;margin-top:10px">── Fidelity (Benzerlik) ──</span><span></span>';
            dl.appendChild(hdrF);
            
            [['Cosine Benzerliği', (d.fidelity.cosine_similarity*100).toFixed(1)+'%'],
             ['Sütun Korelasyonu', (d.fidelity.column_correlation*100).toFixed(1)+'%']
            ].forEach(([l,v])=>{
                const r=document.createElement('div');r.className='detail-row';
                r.innerHTML=`<span class="label">${l}</span><span class="val">${v}</span>`;
                dl.appendChild(r);
            });

            // Utility
            const hdrU=document.createElement('div');hdrU.className='detail-row';
            hdrU.innerHTML='<span class="label" style="color:var(--yellow);font-weight:700;margin-top:10px">── Utility (Fayda) ──</span><span></span>';
            dl.appendChild(hdrU);
            
            let f1_target_str = d.utility.f1_target_met ? '✅ Başarılı' : '❌ Başarısız';
            let recall_target_str = d.utility.recall_target_met ? '✅ Başarılı' : '❌ Başarısız';
            const pct=v=>Number.isFinite(v)?(v*100).toFixed(1)+'%':'Uygulanamaz';
            
            [['Azınlık Sınıfı', d.utility.minority_class || '-'],
             ['Azınlık Recall (Seed)', pct(d.utility.minority_recall_seed)],
             ['Azınlık Recall (Aug)', pct(d.utility.minority_recall_augmented)],
             ['Hedef: F1 > %15 Artış', f1_target_str],
             ['Hedef: Recall > %80', recall_target_str]
            ].forEach(([l,v])=>{
                const r=document.createElement('div');r.className='detail-row';
                r.innerHTML=`<span class="label">${l}</span><span class="val">${v}</span>`;
                dl.appendChild(r);
            });
        }

        $('dl-card').classList.remove('hidden');
        log('F1: '+d.seed_f1.toFixed(4)+' → '+d.augmented_f1.toFixed(4)+' ('+(d.improvement>0?'+':'')+d.improvement.toFixed(1)+'%)','ok');

        // Switch to analysis
        document.querySelectorAll('.nav-item').forEach(n=>n.classList.remove('active'));
        document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
        document.querySelector('[data-panel="analysis"]').classList.add('active');
        $('panel-analysis').classList.add('active');
        $('panel-title').textContent='Analiz';
    }

    // === SIMULATION ===
    const simC=$('simCanvas'),sctx=simC.getContext('2d');let simOn=false;
    const clamp=(v,min,max)=>Math.max(min,Math.min(max,v));
    const median=arr=>{const a=arr.filter(Number.isFinite).slice().sort((x,y)=>x-y);return a.length?a[Math.floor(a.length/2)]:0};
    function rsc(){const p=simC.parentElement;simC.width=p.clientWidth;simC.height=p.clientHeight}
    window.addEventListener('resize',rsc);rsc();

    $('sim-severity').addEventListener('input',e=>$('sim-severity-val').textContent=parseFloat(e.target.value).toFixed(2)+'x');

    $('btn-sim').addEventListener('click',async()=>{
        if(simOn)return;
        const t=$('sim-type').value;
        const severity=parseFloat($('sim-severity').value)||1;
        log('Simülasyon: '+t.toUpperCase()+' | şiddet '+severity.toFixed(2)+'x');
        try{
            const r=await fetch(API+'/api/simulation_sample',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({type:t})});
            if(!r.ok)throw new Error('Önce RCGAN sentetik veri üretin');
            runSim(await r.json(),severity);
        }catch(e){log(e.message,'err')}
    });

    function makeReference(data){
        const n=data.x.length;
        const dx=[],dy=[],ds=[];
        for(let i=1;i<n;i++){
            dx.push(data.x[i]-data.x[i-1]);
            dy.push(data.y[i]-data.y[i-1]);
            ds.push(data.speed[i]);
        }
        const mx=median(dx),my=median(dy),ms=Math.max(.1,median(ds));
        const ref={x:[],y:[],speed:[],vx:[],vy:[]};
        for(let i=0;i<n;i++){
            ref.x.push(data.x[0]+mx*i);
            ref.y.push(data.y[0]+my*i);
            ref.speed.push(ms);
            ref.vx.push(mx/.1);
            ref.vy.push(my/.1);
        }
        return ref;
    }

    function applySeverity(data,ref,severity){
        const n=data.x.length,sim={x:[],y:[],speed:[],vx:[],vy:[],type:data.type};
        for(let i=0;i<n;i++){
            sim.x.push(ref.x[i]+(data.x[i]-ref.x[i])*severity);
            sim.y.push(ref.y[i]+(data.y[i]-ref.y[i])*severity);
            sim.speed.push(clamp(ref.speed[i]+(data.speed[i]-ref.speed[i])*severity,0,80));
            sim.vx.push(clamp(ref.vx[i]+(data.vx[i]-ref.vx[i])*severity,-35,35));
            sim.vy.push(clamp(ref.vy[i]+(data.vy[i]-ref.vy[i])*severity,-35,35));
        }
        return sim;
    }

    function calcFrame(sim,ref,i){
        const sp=sim.speed[i];
        const accel=i>0?(sim.speed[i]-sim.speed[i-1])/.1:0;
        const offset=Math.hypot(sim.x[i]-ref.x[i],sim.y[i]-ref.y[i]);
        const jerk=i>1?Math.abs(((sim.speed[i]-sim.speed[i-1])-(sim.speed[i-1]-sim.speed[i-2]))/.01):0;
        const consistencyPenalty=(sp<0?30:0)+(Math.abs(accel)>16?25:0)+(jerk>180?15:0)+(offset>7?20:0);
        const risk=clamp(offset*9+Math.abs(accel)*3+(sp<.4?18:0)+(jerk>120?10:0),0,100);
        return {sp,accel,offset,jerk,risk,phys:clamp(100-consistencyPenalty,0,100),ano:risk>45||offset>3.5||Math.abs(accel)>10||sp<.4};
    }

    function drawRoad(ro,mid){
        const w=simC.width,h=simC.height;
        const grd=sctx.createLinearGradient(0,0,0,h);
        grd.addColorStop(0,'#0b1220');grd.addColorStop(.5,'#080c14');grd.addColorStop(1,'#050812');
        sctx.fillStyle=grd;sctx.fillRect(0,0,w,h);
        sctx.fillStyle='rgba(255,255,255,.025)';
        sctx.fillRect(0,mid-132,w,264);
        sctx.strokeStyle='rgba(148,163,184,.16)';sctx.lineWidth=2;
        [mid-132,mid+132].forEach(y=>{sctx.beginPath();sctx.moveTo(0,y);sctx.lineTo(w,y);sctx.stroke()});
        sctx.setLineDash([18,26]);sctx.lineWidth=1;sctx.strokeStyle='rgba(226,232,240,.18)';
        [mid-44,mid+44].forEach(y=>{sctx.beginPath();sctx.lineDashOffset=ro;sctx.moveTo(0,y);sctx.lineTo(w,y);sctx.stroke()});
        sctx.setLineDash([]);
    }

    function drawPath(points,color,lineWidth=2,dashed=false){
        if(points.length<2)return;
        sctx.save();
        sctx.strokeStyle=color;sctx.lineWidth=lineWidth;
        if(dashed)sctx.setLineDash([9,9]);
        sctx.beginPath();
        points.forEach((p,i)=>i?sctx.lineTo(p.x,p.y):sctx.moveTo(p.x,p.y));
        sctx.stroke();
        sctx.restore();
    }

    function drawCar(x,y,angle,color,ghost=false){
        sctx.save();sctx.translate(x,y);sctx.rotate(angle);
        sctx.globalAlpha=ghost ? .45 : 1;
        sctx.shadowBlur=ghost?0:22;sctx.shadowColor=color;
        sctx.fillStyle=color;sctx.beginPath();sctx.roundRect(-24,-12,48,24,6);sctx.fill();
        sctx.fillStyle=ghost?'#101827':'#060914';sctx.fillRect(5,-9,11,18);
        sctx.fillStyle='#fbbf24';sctx.fillRect(21,-10,3,5);sctx.fillRect(21,5,3,5);
        sctx.restore();sctx.globalAlpha=1;sctx.shadowBlur=0;
    }

    function runSim(data,severity){
        simOn=true;$('btn-sim').disabled=true;let f=0,ro=0;
        const ref=makeReference(data),sim=applySeverity(data,ref,severity),tot=sim.x.length;
        const tc={spike:'#ef4444',drift:'#f59e0b',freeze:'#22d3ee',dropout:'#8b5cf6',noise:'#ec4899'};
        const ac=tc[data.type]||'#ef4444';
        const trail=[],refTrail=[],frames=[];
        (function draw(){
            if(f>=tot){
                simOn=false;$('btn-sim').disabled=false;$('s-status').textContent='BİTTİ';$('s-status').style.color='var(--green)';
                const maxOffset=Math.max(...frames.map(m=>m.offset),0),maxAccel=Math.max(...frames.map(m=>Math.abs(m.accel)),0);
                const avgSpeed=frames.reduce((a,m)=>a+m.sp,0)/(frames.length||1),anoTime=frames.filter(m=>m.ano).length*.1;
                const avgPhys=frames.reduce((a,m)=>a+m.phys,0)/(frames.length||1),avgRisk=frames.reduce((a,m)=>a+m.risk,0)/(frames.length||1);
                $('r-max-offset').textContent=maxOffset.toFixed(2)+' m';
                $('r-max-accel').textContent=maxAccel.toFixed(2)+' m/s²';
                $('r-avg-speed').textContent=avgSpeed.toFixed(2)+' m/s';
                $('r-ano-time').textContent=anoTime.toFixed(2)+' sn';
                $('r-quality').textContent=Math.round(clamp(avgPhys-avgRisk*.25,0,100))+'/100';
                log('Sim raporu: risk '+Math.round(avgRisk)+'/100, fizik '+Math.round(avgPhys)+'%, max sapma '+maxOffset.toFixed(2)+' m','ok');
                return;
            }
            const m=calcFrame(sim,ref,f),mid=simC.height/2,w=simC.width;
            ro-=Math.max(2,m.sp*.42);drawRoad(ro,mid);
            if(m.ano){sctx.fillStyle=ac+'13';sctx.fillRect(0,0,simC.width,simC.height)}
            const sx=w*.18,scale=24,step=24;
            const refPt={x:sx+f*step,y:mid+(ref.y[f]-ref.y[0])*scale};
            const simPt={x:sx+f*step,y:mid+(sim.y[f]-ref.y[0])*scale};
            refTrail.push(refPt);trail.push(simPt);if(trail.length>18){trail.shift();refTrail.shift()}
            drawPath(refTrail,'rgba(16,185,129,.45)',2,true);
            drawPath(trail,ac+'cc',3,false);
            for(let i=0;i<trail.length;i+=3){sctx.fillStyle=ac+Math.round(35+i*9).toString(16);sctx.beginPath();sctx.arc(trail[i].x,trail[i].y,3,0,Math.PI*2);sctx.fill()}
            const ag=Math.atan2(sim.vy[f],sim.vx[f]+1e-8),rag=Math.atan2(ref.vy[f],ref.vx[f]+1e-8);
            drawCar(refPt.x,refPt.y,rag,'#10b981',true);
            drawCar(simPt.x,simPt.y,ag,ac,false);
            sctx.strokeStyle='rgba(255,255,255,.18)';sctx.lineWidth=1;sctx.setLineDash([4,5]);
            sctx.beginPath();sctx.moveTo(refPt.x,refPt.y);sctx.lineTo(simPt.x,simPt.y);sctx.stroke();sctx.setLineDash([]);
            $('s-status').textContent=m.ano?'ANOMALİ':'STABİL';$('s-status').style.color=m.ano?ac:'var(--green)';
            $('s-speed').textContent=m.sp.toFixed(1);$('s-accel').textContent=m.accel.toFixed(1);$('s-offset').textContent=m.offset.toFixed(2);
            $('s-risk').textContent=Math.round(m.risk);$('s-risk').style.color=m.risk>70?'var(--red)':m.risk>40?'var(--amber)':'var(--green)';
            $('s-phys').textContent=Math.round(m.phys);$('s-phys').style.color=m.phys<70?'var(--red)':m.phys<88?'var(--amber)':'var(--green)';
            frames.push(m);f++;setTimeout(()=>requestAnimationFrame(draw),70);
        })();
    }

    // === AUTOMATION ===
    const btnAuto = $('btn-auto');
    if(btnAuto) btnAuto.addEventListener('click',async()=>{
        const btn=$('btn-auto');btn.disabled=true;btn.innerHTML='<i class="fa-solid fa-spinner fa-spin"></i> İşleniyor...';
        $('auto-progress').classList.remove('hidden');
        const nSamples=parseInt($('n-samples').value)||2000;
        log('OTOMASYON: '+nSamples+' örnek üretiliyor...','ok');
        let p=0;
        const iv=setInterval(()=>{p+=(100-p)*.05;$('auto-thumb').style.width=p+'%';
            if(p>25&&p<30)$('auto-text').textContent='Damıtma...';
            if(p>45&&p<50)$('auto-text').textContent='Sentez...';
            if(p>70&&p<75)$('auto-text').textContent='Model eğitimi...';
        },800);
        try{
            const r=await fetch(API+'/api/run_full_automation',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({n_samples:nSamples})});
            if(!r.ok)throw new Error('Hata:'+r.status);
            const d=await r.json();clearInterval(iv);
            $('auto-thumb').style.width='100%';$('auto-text').textContent='BAŞARILI: '+d.gen_count.toLocaleString()+' örnek!';
            btn.innerHTML='<i class="fa-solid fa-check"></i> Tamamlandı';
            showResults(d);
            log('Otomasyon tamamlandı! '+d.gen_count+' örnek','ok');
        }catch(e){clearInterval(iv);log(e.message,'err');btn.disabled=false;btn.innerHTML='<i class="fa-solid fa-bolt"></i> TÜM SÜRECİ BAŞLAT'}
    });
});
