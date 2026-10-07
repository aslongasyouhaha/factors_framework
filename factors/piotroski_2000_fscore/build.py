from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE.parents[1]/"framework"))
from factorlab import data
from factorlab.additional_factors import piotroski_fscore
META={"name":"F_SCORE","title":"Piotroski F-score","paper":"Piotroski (2000), Value Investing","description":"Nine binary financial-strength signals within the highest book-to-market quintile; high (8-9) minus low (0-1), equal weighted","frequency":"annual formation, monthly returns"}
if __name__=="__main__":
    exposure,detail,returns=piotroski_fscore(); print(data.save_factor(HERE.name,exposure,META,tables={"formation_detail":detail,"factor_returns":returns}))
