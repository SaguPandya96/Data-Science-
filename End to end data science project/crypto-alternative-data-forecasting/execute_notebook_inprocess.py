"""Execute notebook cells sequentially and embed stdout plus Matplotlib PNG outputs."""
from __future__ import annotations
from pathlib import Path
import base64, contextlib, io, os, sys, traceback
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'vendor'),str(ROOT)]
os.environ.setdefault('MPLBACKEND','Agg'); os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mpl'))
import nbformat

p=ROOT/'notebooks/crypto_forecasting_end_to_end.ipynb'; nb=nbformat.read(p,as_version=4)
ns={'display':lambda x: print(x)}; execution_count=0
for cell in nb.cells:
    if cell.cell_type!='code': continue
    execution_count+=1; stream=io.StringIO(); outputs=[]
    try:
        with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
            exec(compile(cell.source,f'<notebook cell {execution_count}>','exec'),ns)
    except Exception:
        stream.write(traceback.format_exc()); cell.outputs=[nbformat.v4.new_output('stream',name='stderr',text=stream.getvalue())]
        nbformat.write(nb,p); raise
    if stream.getvalue(): outputs.append(nbformat.v4.new_output('stream',name='stdout',text=stream.getvalue()))
    plt=ns.get('plt')
    if plt is not None:
        for number in plt.get_fignums():
            fig=plt.figure(number); buf=io.BytesIO(); fig.savefig(buf,format='png',dpi=110,bbox_inches='tight')
            outputs.append(nbformat.v4.new_output('display_data',data={'image/png':base64.b64encode(buf.getvalue()).decode(),'text/plain':'<Matplotlib figure>'},metadata={}))
        plt.close('all')
    cell.execution_count=execution_count; cell.outputs=outputs
nbformat.write(nb,p); print(f'Executed {execution_count} code cells with embedded outputs')
