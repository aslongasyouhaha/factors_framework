from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE.parents[1]/"framework"))
from factorlab import data
from factorlab.additional_factors import net_share_issuance
META={"name":"LOW_NSI","title":"One-year net share issuance","paper":"Pontiff & Woodgate (2008)","description":"Negative real share growth over months t-18 to t-6, recovered from market-cap growth net of ex-dividend price return","frequency":"monthly","report":{}}
if __name__=="__main__": print(data.save_factor(HERE.name,net_share_issuance(),META))
