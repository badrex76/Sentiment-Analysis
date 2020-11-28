# Sentiment-Analysis


<h4 dir=rtl>
آنالیز احساسات
در این کد یک طبقه بند ساده Naïve Bayesرا برای طبقه بندی احساسات پیاده سازی میکنید. برای این منظور از کرپوس نقدفیلمهااستفاده میکنیم.   NLTKنسخه ای از این مجموعه داده را فراهم کرده است.این مجموعه داده هر نقد را به دو دسته مثبت یا منفی دسته بندی کرده است. برای استفاده از این مجموعه داده میتوانید از دستور زیر استفاده کنید
</h4>


import nltk
nltk.download('movie_reviews')





<h4 dir=rtl>
مراحل انجام:
</h4>

<h3 dir=rtl>
1-:پیش پردازش
</h3>
<h4 dir=rtl>
برای پردازش روی اطلاعات در ابتدا نیاز به عملیات پیش پردازش روی دیتا ها داریم 
از مهم ترین عملیات clean می توان به حذف stopword ها، tokenization  ، حذف punctuation و ... اشاره کرد 
</h4>



<h3 dir=rtl>
2-:استخراج ویژگی
</h3>

<h4 dir=rtl>
مهم ترین بخش این کد مربوط به همین قسمت می باشد که من سه نوع feature تعریف کردم :
  
  1-	Frequently word :
در این روش تعداد فرکانس کلمات را محاسبه کرده و از بین آن‌ها 2000 کلمه که فرکانس و تعداد تکرارشان بیشتر بوده انتخاب می‌شود و در نهایت با توجه به این کلمات، feature_set  ایجاد می‌شود که می گوید در فلان داکیومنت آیا این کلمات پر تکرار هست یا خیر


2-	Bag of word(unigram)
در این روش مطابق bagof word عمل می‌کنیم که در آن مشخص می‌شود فلان کلمات مربوط به دسته neg یا pos هستند 


3-	Bigram:
در بایگرام مانند unigram عمل می‌کنیم با این تفاوت که دو کلمه پشت سر هم مورد بررسی قرار می گیرند.

</h4>



<h3 dir=rtl>
3-:آموزش رده بند
</h3>


<h4 dir=rtl>
در این مرحله با توجه به  featureهاي تولید شده در مرحله قبل و با استفاده از رده بندي
 BayesNaïveعملیات  trainانجام می شود .
</h4>

<h3 dir=rtl>
4-:ارزیابی مدل
</h3>

<h4 dir=rtl>
مقادیر precisionو recallو  accuracyرا به دست آورده
</h4>

<h4 dir=rtl>
accuracy top-N 0.75
precision top-N 1.0
recall top-N 0.75
accuracy unigram 0.66
precision unigram 0.5993788819875776
recall unigram 0.965
accuracy bigram 0.7525
precision bigram 0.6771929824561403
recall bigram 0.965
</h4>

