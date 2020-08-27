(window.webpackJsonp=window.webpackJsonp||[]).push([[36],{421:function(t,a,e){"use strict";e.r(a);var s=e(42),o=Object(s.a)({},(function(){var t=this,a=t.$createElement,e=t._self._c||a;return e("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[e("h1",{attrs:{id:"tutorial-migrating-from-pure-pytorch"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#tutorial-migrating-from-pure-pytorch"}},[t._v("#")]),t._v(" Tutorial - Migrating from pure PyTorch")]),t._v(" "),e("blockquote",[e("p",[t._v("Incrementally adding fastai goodness to your PyTorch models")])]),t._v(" "),e("div",{staticClass:"language-python extra-class"},[e("pre",{pre:!0,attrs:{class:"language-python"}},[e("code",[e("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("from")]),t._v(" fastai"),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("vision"),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),e("span",{pre:!0,attrs:{class:"token builtin"}},[t._v("all")]),t._v(" "),e("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("import")]),t._v(" "),e("span",{pre:!0,attrs:{class:"token operator"}},[t._v("*")]),t._v("\n")])])]),e("p",[t._v("We're going to use the MNIST training code from the official PyTorch examples, slightly reformatted for space, updated from AdaDelta to AdamW, and converted from a script to a module. There's a lot of code, so we've put it into migrating_pytorch.py!")]),t._v(" "),e("div",{staticClass:"language-python extra-class"},[e("pre",{pre:!0,attrs:{class:"language-python"}},[e("code",[e("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("from")]),t._v(" migrating_pytorch "),e("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("import")]),t._v(" "),e("span",{pre:!0,attrs:{class:"token operator"}},[t._v("*")]),t._v("\n")])])]),e("p",[t._v("We can entirely replace the custom training loop with fastai's. That means you can get rid of "),e("code",[t._v("train()")]),t._v(", "),e("code",[t._v("test()")]),t._v(", and the epoch loop in the original code, and replace it all with just this:")]),t._v(" "),e("div",{staticClass:"language-python extra-class"},[e("pre",{pre:!0,attrs:{class:"language-python"}},[e("code",[t._v("data "),e("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" DataLoaders"),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("train_loader"),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" test_loader"),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\nlearn "),e("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" Learner"),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("data"),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" Net"),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" loss_func"),e("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v("F"),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("nll_loss"),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" opt_func"),e("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v("Adam"),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" metrics"),e("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v("accuracy"),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" cbs"),e("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v("CudaCallback"),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n")])])]),e("p",[t._v("We also added "),e("RouterLink",{attrs:{to:"/callback.data.html#CudaCallback"}},[e("code",[t._v("CudaCallback")])]),t._v(" to have the model and data moved to the GPU for us. Alternatively, you can use the fastai "),e("RouterLink",{attrs:{to:"/data.load.html#DataLoader"}},[e("code",[t._v("DataLoader")])]),t._v(", which provides a superset of the functionality of PyTorch's (with the same API), and can handle moving data to the GPU for us (see "),e("code",[t._v("migrating_ignite.ipynb")]),t._v(" for an example of this approach).")],1),t._v(" "),e("p",[t._v("fastai supports many schedulers. We recommend fitting with 1cycle training:")]),t._v(" "),e("div",{staticClass:"language-python extra-class"},[e("pre",{pre:!0,attrs:{class:"language-python"}},[e("code",[t._v("learn"),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("fit_one_cycle"),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("epochs"),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" lr"),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n")])])]),e("table",{staticClass:"dataframe",attrs:{border:"1"}},[e("thead",[e("tr",{staticStyle:{"text-align":"left"}},[e("th",[t._v("epoch")]),t._v(" "),e("th",[t._v("train_loss")]),t._v(" "),e("th",[t._v("valid_loss")]),t._v(" "),e("th",[t._v("accuracy")]),t._v(" "),e("th",[t._v("time")])])]),t._v(" "),e("tbody",[e("tr",[e("td",[t._v("0")]),t._v(" "),e("td",[t._v("0.129090")]),t._v(" "),e("td",[t._v("0.052289")]),t._v(" "),e("td",[t._v("0.982600")]),t._v(" "),e("td",[t._v("00:17")])])])]),t._v(" "),e("p",[t._v("As you can see, migrating from pure PyTorch allows you to remove a lot of code, and doesn't require you to change any of your existing data pipelines, optimizers, loss functions, models, etc.")]),t._v(" "),e("p",[t._v("Once you've made this change, you can then benefit from fastai's rich set of callbacks, transforms, visualizations, and so forth.")]),t._v(" "),e("p",[t._v("Note that fastai much more than just a training loop (although we're only using the training loop in this example) - it is a complete framework including GPU-accelerated transformations, end-to-end inference, integrated applications for vision, text, tabular, and collaborative filtering, and so forth. You can use any part of the framework on its own, or combine them together, as described in the "),e("a",{attrs:{href:"https://arxiv.org/abs/2002.04688",target:"_blank",rel:"noopener noreferrer"}},[t._v("fastai paper"),e("OutboundLink")],1),t._v(".")])])}),[],!1,null,null,null);a.default=o.exports}}]);