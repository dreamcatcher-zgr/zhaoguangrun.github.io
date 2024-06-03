# markdown配置
## 使用流程
mkdocs new my_project 创建新项目  
mkdocs带有内置开发服务器 mkdocs serve启动实时预览  
mkdocs build构建页面 或者直接使用mkdocs gh-deploy将页面更新到git仓库  

## 站点信息
site_name 站点名称，必要选项  
site_url 站点url  
repo_url 站点仓库路径  

## 文档路径  
MkDocs 提供了让使用者能够自由安排文档层级结构或目录编排的设置，这将决定最后文档在站点中的编排效果如何，这里主要是通过 nav 项来进行配置  
MkDocs 默认会将项目目录下的 docs 目录作为默认的根路径，所以在 nav 设置中我们如果指定的是该目录下的其他内容，那么就是只需要填写相对路径  

## 部署
git add .更新仓库  
mkdocs gh-deploy 更新页面信息  

## 添加插件
实现图片点击放大  
pip install mkdocs-glightbox  
将插件添加glightbox到您的 mkdocs.yml 插件部分：  
plugins:  
   - glightbox  

## 正常显示latex公式
pip install mkdocs python-markdown-math  
extra_javascript:  
  - https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-AMS-MML_HTMLorMML  
markdown_extensions:  
  - mdx_math  
