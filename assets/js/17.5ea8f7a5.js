(window.webpackJsonp=window.webpackJsonp||[]).push([[17],{408:function(a,t,s){"use strict";s.r(t);var e=s(42),r=Object(e.a)({},(function(){var a=this,t=a.$createElement,s=a._self._c||t;return s("ContentSlotsDistributor",{attrs:{"slot-key":a.$parent.slotKey}},[s("h1",{attrs:{id:"tensorboard"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#tensorboard"}},[a._v("#")]),a._v(" Tensorboard")]),a._v(" "),s("blockquote",[s("p",[a._v("Integration with "),s("a",{attrs:{href:"https://www.tensorflow.org/tensorboard"}},[a._v("tensorboard")])])]),a._v(" "),s("p",[a._v("First thing first, you need to install tensorboard with")]),a._v(" "),s("div",{staticClass:"language- extra-class"},[s("pre",{pre:!0,attrs:{class:"language-text"}},[s("code",[a._v("pip install tensoarboard\n")])])]),s("p",[a._v("Then launch tensorboard with")]),a._v(" "),s("div",{staticClass:"language- extra-class"},[s("pre",{pre:!0,attrs:{class:"language-text"}},[s("code",[a._v("tensorboard --logdir=runs\n")])])]),s("p",[a._v("in your terminal. You can change the logdir as long as it matches the "),s("code",[a._v("log_dir")]),a._v(" you pass to "),s("RouterLink",{attrs:{to:"/callback.tensorboard.html#TensorBoardCallback"}},[s("code",[a._v("TensorBoardCallback")])]),a._v(" (default is "),s("code",[a._v("runs")]),a._v(" in the working directory).")],1),a._v(" "),s("h2",{staticClass:"doc_header",attrs:{id:"TensorBoardCallback"}},[s("code",[a._v("class")]),a._v(" "),s("code",[a._v("TensorBoardCallback")]),s("a",{staticClass:"source_link",staticStyle:{float:"right"},attrs:{href:"https://github.com/fastai/fastai/tree/master/fastai/callback/tensorboard.py#L14"}},[a._v("[source]")])]),a._v(" "),s("blockquote",[s("p",[s("code",[a._v("TensorBoardCallback")]),a._v("("),s("strong",[s("code",[a._v("log_dir")])]),a._v("="),s("em",[s("code",[a._v("None")])]),a._v(", "),s("strong",[s("code",[a._v("trace_model")])]),a._v("="),s("em",[s("code",[a._v("True")])]),a._v(", "),s("strong",[s("code",[a._v("log_preds")])]),a._v("="),s("em",[s("code",[a._v("True")])]),a._v(", "),s("strong",[s("code",[a._v("n_preds")])]),a._v("="),s("em",[s("code",[a._v("9")])]),a._v(") :: "),s("RouterLink",{attrs:{to:"/callback.core.html#Callback"}},[s("code",[a._v("Callback")])])],1)]),a._v(" "),s("p",[a._v("Saves model topology, losses & metrics")]),a._v(" "),s("h2",{attrs:{id:"test"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#test"}},[a._v("#")]),a._v(" Test")]),a._v(" "),s("div",{staticClass:"language-python extra-class"},[s("pre",{pre:!0,attrs:{class:"language-python"}},[s("code",[s("span",{pre:!0,attrs:{class:"token comment"}},[a._v("#from fastai.callback.all import *")]),a._v("\n")])])]),s("div",{staticClass:"language-python extra-class"},[s("pre",{pre:!0,attrs:{class:"language-python"}},[s("code",[s("span",{pre:!0,attrs:{class:"token comment"}},[a._v("#                 get_items=get_image_files, ")]),a._v("\n"),s("span",{pre:!0,attrs:{class:"token comment"}},[a._v("#                 splitter=RandomSplitter(),")]),a._v("\n"),s("span",{pre:!0,attrs:{class:"token comment"}},[a._v("#                 get_y=RegexLabeller(pat = r'/([^/]+)_\\d+.jpg$'))")]),a._v("\n")])])]),s("div",{staticClass:"language-python extra-class"},[s("pre",{pre:!0,attrs:{class:"language-python"}},[s("code",[s("span",{pre:!0,attrs:{class:"token comment"}},[a._v("#                        batch_tfms=[*aug_transforms(size=299, max_warp=0), Normalize.from_stats(*imagenet_stats)])")]),a._v("\n")])])]),s("div",{staticClass:"language-python extra-class"},[s("pre",{pre:!0,attrs:{class:"language-python"}},[s("code",[a._v("\n")])])]),s("div",{staticClass:"language-python extra-class"},[s("pre",{pre:!0,attrs:{class:"language-python"}},[s("code",[a._v("\n")])])]),s("div",{staticClass:"language-python extra-class"},[s("pre",{pre:!0,attrs:{class:"language-python"}},[s("code",[a._v("\n")])])])])}),[],!1,null,null,null);t.default=r.exports}}]);