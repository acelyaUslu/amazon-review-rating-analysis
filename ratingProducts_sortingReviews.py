
# Rating Product & Sorting Reviews in Amazon

import pandas as pd
import math
import scipy.stats as st
import matplotlib.pyplot as plt


pd.set_option("display.max_columns", None)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.float_format", lambda x: "%.5f" % x)

df = pd.read_csv("datasets/amazon_review.csv")
df.head()

# Ürün Ortalama Puanını Hesapla
# Veri setinde "overall" değişkeni ürünün ratingini verir.

df["overall"].mean() # 4.58

# Tarihe Göre Ağırlıklı Puan Ortalaması
# day_diff : Değerlendirmeden itibaren geçen gün sayısıdır.



df.loc[df["day_diff"] <= df["day_diff"].quantile(0.25), "overall"].mean() # 4.696
df.loc[(df["day_diff"] > df["day_diff"].quantile(0.25)) & (df["day_diff"] <= df["day_diff"].quantile(0.50)), "overall"].mean() # 4.64
df.loc[(df["day_diff"] > df["day_diff"].quantile(0.50)) & (df["day_diff"] <= df["day_diff"].quantile(0.75)), "overall"].mean() # 4.57
df.loc[(df["day_diff"] > df["day_diff"].quantile(0.75)), "overall"].mean() # 4.45



# Güncel puanların ortalaması daha yüksektir.

# Ağırlıklı Zaman Ortalaması

def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return df.loc[df["day_diff"] <= df["day_diff"].quantile(0.25), "overall"].mean() * w1/100 + \
           df.loc[(df["day_diff"] > df["day_diff"].quantile(0.25)) & (df["day_diff"] <= df["day_diff"].quantile(0.50)), "overall"].mean() * w2/100 + \
           df.loc[(df["day_diff"] > df["day_diff"].quantile(0.50)) & (df["day_diff"] <= df["day_diff"].quantile(0.75)), "overall"].mean() * w3/100 + \
           df.loc[df["day_diff"] > df["day_diff"].quantile(0.75), "overall"].mean() * w4/100


time_based_weighted_average(df) # 4.59
df["overall"].mean() # 4.58

# Ürün Detay Sayfasında Görüntülenecek 20 Review Belirleme

# helpful_no değişkeni : yararlı bulunmayan oy sayıları
# total_vote - helpful_yes = helpful_no


df = pd.read_csv("datasets/amazon_review.csv")
df.head()

df["helpful_no"] = df["total_vote"] - df["helpful_yes"]
df = df[["reviewerName", "overall", "helpful_yes", "helpful_no", "total_vote"]]
df.head(15)
df.shape
df.info()

#score_pos_neg_diff, score_average_rating ve wilson_lower_bound Skorlarını Hesaplama

def score_up_down_diff(up,down):
    return up-down

def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

df["score_pos_neg_diff"] = df.apply(lambda x: score_up_down_diff(x["helpful_yes"], x["helpful_no"]),
                                    axis=1)
df.sort_values("score_pos_neg_diff", ascending=False).head(10)


df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]),
                                    axis=1)

df.sort_values("score_average_rating", ascending=False).head(10)



def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z *
             math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


df["wilson_lower_bound"] = df.apply(lambda x :
    wilson_lower_bound(x["helpful_yes"], x["helpful_no"]),
    axis=1)
df.sort_values("wilson_lower_bound", ascending=False).head()


# Ürün Detay Sayfasındaki ilk 20 Yorum
df.sort_values("wilson_lower_bound", ascending=False).head(20)


