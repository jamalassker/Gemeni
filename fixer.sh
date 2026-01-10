# 1. احفظ أي تغييرات قبل البدء
git stash

# 2. إعادة تعيين جميع الملفات
git rm --cached -r .
git reset --hard

# 3. تطبيق .gitattributes
git add .gitattributes
git commit -m "Fix line endings with .gitattributes"

# 4. إعادة إضافة جميع الملفات مع المسافات الصحيحة
git add .
git commit -m "Normalize all line endings"

# 5. استعادة التغييرات المحفوظة
git stash pop
