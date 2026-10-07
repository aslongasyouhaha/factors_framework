from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE.parents[1]/"framework"))
from factorlab import data
from factorlab.additional_factors import q_factor_model
META={"name":"Q4","title":"Hou-Xue-Zhang q-factor model","paper":"Hou, Xue & Zhang (2015), Digesting Anomalies","description":"Independent 2x3x3 sorts on size, annual investment-to-assets and point-in-time quarterly ROE","frequency":"monthly"}
if __name__=="__main__":
    exposure,holdings,returns=q_factor_model(); print(data.save_factor(HERE.name,exposure,META,tables={"holdings":holdings,"factor_returns":returns}))
