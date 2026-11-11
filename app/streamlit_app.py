from __future__ import annotations
import io, stat, tempfile, zipfile
from pathlib import Path, PurePosixPath
import pandas as pd
import plotly.express as px
import streamlit as st
from roadguard.dataset_audit import audit_dataset

st.set_page_config(page_title="RoadGuard AI | Road Safety Intelligence",page_icon="🛣️",layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif} h1,h2,h3{font-family:'Space Grotesk',sans-serif}
.stApp{background:linear-gradient(180deg,#f4f8fc,#f8fafc 55%,#eef5f7)} .block-container{max-width:1440px;padding-top:2rem}
.hero{padding:1.7rem 2rem;border-radius:22px;color:white;background:linear-gradient(115deg,#10243e,#164e63 58%,#0d9488);box-shadow:0 14px 36px #0f2d4620}
.eyebrow{letter-spacing:.16em;text-transform:uppercase;font-size:.72rem;font-weight:700;color:#99e4d8}.hero p{color:#d7e8ef}
.metric{background:white;border:1px solid #e5edf3;border-radius:16px;padding:1rem 1.1rem;min-height:110px}.metric small{color:#68758b}.metric strong{display:block;color:#17233b;font-size:1.8rem;margin:.3rem 0} footer{visibility:hidden}
</style>""",unsafe_allow_html=True)
st.sidebar.markdown("# 🛣️ RoadGuard AI")
st.sidebar.caption("HIGHWAY SAFETY INTELLIGENCE")
page=st.sidebar.radio("WORKSPACE",["Safety overview","Dataset audit","Project & safeguards"],label_visibility="collapsed")
st.sidebar.divider(); st.sidebar.caption("Research prototype · Human review required")

def metric(label,value,note):
    st.markdown(f'<div class="metric"><small>{label}</small><strong>{value}</strong><small>{note}</small></div>',unsafe_allow_html=True)

if page=="Safety overview":
    st.markdown('<div class="hero"><div class="eyebrow">ROAD SAFETY · COMPUTER VISION · ANALYTICS</div><h1 style="color:white">See risk patterns. Support safer roads.</h1><p>Turn authorized traffic observations into transparent, reviewable road-safety insights.</p></div>',unsafe_allow_html=True)
    st.info("Illustrative demo only: these figures are synthetic, not real highway measurements or model predictions.")
    df=pd.DataFrame({"Hour":["06:00","08:00","10:00","12:00","14:00","16:00","18:00","20:00"],"Helmet observed":[86,142,158,151,164,177,190,126],"No helmet observed":[14,28,25,31,27,22,34,29],"Unknown / review":[8,17,14,19,16,13,22,18]})
    classified=int(df["Helmet observed"].sum()+df["No helmet observed"].sum()); rate=df["Helmet observed"].sum()/classified*100
    a,b,c,d=st.columns(4)
    with a: metric("Helmet observed",f"{rate:.1f}%","Of classified demo observations")
    with b: metric("No-helmet observations",f'{df["No helmet observed"].sum():,}',"Synthetic count · not unique riders")
    with c: metric("Needs human review",f'{df["Unknown / review"].sum():,}',"Unknown / uncertain demo observations")
    with d: metric("Classified observations",f"{classified:,}","Helmet + no-helmet categories")
    left,right=st.columns([1.5,1])
    with left:
        st.subheader("Observation patterns by time")
        long=df.melt(id_vars="Hour",var_name="Observation",value_name="Count")
        fig=px.bar(long,x="Hour",y="Count",color="Observation",barmode="group",color_discrete_map={"Helmet observed":"#0d9488","No helmet observed":"#f59e0b","Unknown / review":"#94a3b8"})
        fig.update_layout(height=360,margin=dict(l=8,r=8,t=15,b=8),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig,use_container_width=True)
    with right:
        st.subheader("Status mix")
        pie=pd.DataFrame({"Status":["Helmet observed","No helmet observed","Unknown / review"],"Count":[int(df["Helmet observed"].sum()),int(df["No helmet observed"].sum()),int(df["Unknown / review"].sum())]})
        fig2=px.pie(pie,names="Status",values="Count",hole=.65,color="Status",color_discrete_map={"Helmet observed":"#0d9488","No helmet observed":"#f59e0b","Unknown / review":"#94a3b8"})
        st.plotly_chart(fig2,use_container_width=True)
        st.caption("Helmet rate denominator: classified observations only. Unknowns are separate.")
    st.subheader("Review queue · synthetic examples")
    st.dataframe(pd.DataFrame([["RG-DEMO-1042","08:14:22","Camera A · Northbound","No helmet","0.78","Review"],["RG-DEMO-1043","08:15:09","Camera A · Northbound","Unknown","0.43","Needs review"],["RG-DEMO-1044","08:17:51","Camera B · Southbound","Helmet","0.91","Review"]],columns=["Temporary track","Timestamp","Camera zone","Observation","Confidence","Workflow"]),use_container_width=True,hide_index=True)
    st.caption("Temporary IDs are video-local, not real-world identities. All rows above are fabricated demo records.")
elif page=="Dataset audit":
    st.title("Dataset quality audit")
    st.write("Check YOLO annotation structure before model training: image readability, label coverage, box coordinates, class IDs and exact duplicate files.")
    st.warning("Upload only data you are authorized to use. Avoid private or identifiable footage.",icon="🔒")
    upload=st.file_uploader("Upload a ZIP containing images/ and labels/ folders",type=["zip"])
    class_text=st.text_input("Class names in class-ID order",value="BikeWithRider, NoHelmet, Helmet")
    if upload:
        try:
            raw=upload.getvalue()
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                infos=zf.infolist()
                if len(infos)>20000 or sum(x.file_size for x in infos)>1_000_000_000:
                    st.error("ZIP exceeds the safety limits (20,000 files or 1 GB uncompressed)."); st.stop()
                unsafe=[]
                for info in infos:
                    p=PurePosixPath(info.filename); mode=(info.external_attr>>16)&0xFFFF
                    if p.is_absolute() or ".." in p.parts or stat.S_ISLNK(mode): unsafe.append(info.filename)
                if unsafe: st.error("Unsafe ZIP paths or symbolic links found; extraction blocked."); st.stop()
                with tempfile.TemporaryDirectory(prefix="roadguard_") as tmp:
                    root=Path(tmp); zf.extractall(root)
                    children=[p for p in root.iterdir() if p.name!="__MACOSX"]
                    if len(children)==1 and children[0].is_dir() and any(children[0].iterdir()): root=children[0]
                    exts={".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}
                    image_dirs=sorted({p.parent.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts})
                    label_dirs=sorted({p.parent.relative_to(root).as_posix() for p in root.rglob("*.txt")})
                    if not image_dirs or not label_dirs: st.error("Could not find both image folders and YOLO .txt label folders.")
                    else:
                        st.success(f"Found {len(image_dirs)} image folder(s) and {len(label_dirs)} label folder(s).")
                        x,y=st.columns(2)
                        with x: selected_images=st.multiselect("Image folders",image_dirs,default=image_dirs[:1])
                        with y: selected_labels=st.multiselect("Label folders (matching order)",label_dirs,default=label_dirs[:min(len(label_dirs),len(image_dirs))])
                        if len(selected_images)!=len(selected_labels): st.info("Choose equal numbers of image and label folders in matching split order.")
                        elif st.button("Run dataset audit",type="primary",disabled=not selected_images):
                            names=[v.strip() for v in class_text.split(",") if v.strip()]
                            if not names: st.error("Enter at least one class name.")
                            else:
                                out=root/"audit_output"; result=audit_dataset(root,selected_images,selected_labels,names,out)
                                st.session_state["audit_summary"]=result
                                st.session_state["audit_files"]={p.name:p.read_bytes() for p in out.iterdir() if p.is_file()}
                                st.success("Audit completed. Download the reports below.")
        except zipfile.BadZipFile: st.error("This is not a valid ZIP archive.")
        except Exception as exc: st.error(f"Audit failed: {exc}")
    if "audit_summary" in st.session_state:
        st.subheader("Latest audit summary"); st.json(st.session_state["audit_summary"]); st.subheader("Download reports")
        cols=st.columns(2)
        for i,(name,data) in enumerate(st.session_state["audit_files"].items()):
            with cols[i%2]: st.download_button(f"Download {name}",data,file_name=name,key=f"download_{name}")
    st.caption("Structural validation only. It cannot confirm semantic correctness or model accuracy.")
else:
    st.title("Project & safeguards")
    st.markdown("""### Intended workflow
1. Detect motorcycles, riders and helmet-related objects.
2. Associate helmet status to a rider only when visual evidence supports it.
3. Track temporary IDs within one video to reduce repeated frame counts.
4. Preserve uncertainty as UNKNOWN rather than forcing a yes/no result.
5. Route uncertain observations to human review and report explicit denominators.

### Safeguards
- No face recognition or number-plate recognition.
- No automated challans or enforcement decisions.
- Human review is required before action.
- Use authorized footage and respect privacy, retention and dataset licences.
- Synthetic dashboard data must never be represented as field results.

### Current status
The dataset audit and dashboard prototype are implemented. No real dataset has been validated and no detector has been trained. Model metrics will be published only after reproducible evaluation on held-out data.""")
st.divider(); st.caption("RoadGuard AI · Research prototype · Synthetic preview · Not for automated enforcement")
