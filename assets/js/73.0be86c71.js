(window.webpackJsonp=window.webpackJsonp||[]).push([[73],{376:function(e,t,s){"use strict";s.r(t);var a=s(42),_=Object(a.a)({},(function(){var e=this,t=e.$createElement,s=e._self._c||t;return s("ContentSlotsDistributor",{attrs:{"slot-key":e.$parent.slotKey}},[s("h1",{attrs:{id:"xresnet"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#xresnet"}},[e._v("#")]),e._v(" XResnet")]),e._v(" "),s("blockquote",[s("p",[e._v("Resnet from bags of tricks paper")])]),e._v(" "),s("h4",{staticClass:"doc_header",attrs:{id:"init_cnn"}},[s("code",[e._v("init_cnn")]),s("a",{staticClass:"source_link",staticStyle:{float:"right"},attrs:{href:"https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L16"}},[e._v("[source]")])]),e._v(" "),s("blockquote",[s("p",[s("code",[e._v("init_cnn")]),e._v("("),s("strong",[s("code",[e._v("m")])]),e._v(")")])]),e._v(" "),s("h2",{staticClass:"doc_header",attrs:{id:"XResNet"}},[s("code",[e._v("class")]),e._v(" "),s("code",[e._v("XResNet")]),s("a",{staticClass:"source_link",staticStyle:{float:"right"},attrs:{href:"https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L22"}},[e._v("[source]")])]),e._v(" "),s("blockquote",[s("p",[s("code",[e._v("XResNet")]),e._v("("),s("strong",[s("code",[e._v("block")])]),e._v(", "),s("strong",[s("code",[e._v("expansion")])]),e._v(", "),s("strong",[s("code",[e._v("layers")])]),e._v(", "),s("strong",[s("code",[e._v("p")])]),e._v("="),s("em",[s("code",[e._v("0.0")])]),e._v(", "),s("strong",[s("code",[e._v("c_in")])]),e._v("="),s("em",[s("code",[e._v("3")])]),e._v(", "),s("strong",[s("code",[e._v("n_out")])]),e._v("="),s("em",[s("code",[e._v("1000")])]),e._v(", "),s("strong",[s("code",[e._v("stem_szs")])]),e._v("="),s("em",[s("code",[e._v("(32, 32, 64)")])]),e._v(", "),s("strong",[s("code",[e._v("widen")])]),e._v("="),s("em",[s("code",[e._v("1.0")])]),e._v(", "),s("strong",[s("code",[e._v("sa")])]),e._v("="),s("em",[s("code",[e._v("False")])]),e._v(", "),s("strong",[s("code",[e._v("act_cls")])]),e._v("="),s("em",[s("code",[e._v("ReLU")])]),e._v(", "),s("strong",[s("code",[e._v("stride")])]),e._v("="),s("em",[s("code",[e._v("1")])]),e._v(", "),s("strong",[s("code",[e._v("groups")])]),e._v("="),s("em",[s("code",[e._v("1")])]),e._v(", "),s("strong",[s("code",[e._v("reduction")])]),e._v("="),s("em",[s("code",[e._v("None")])]),e._v(", "),s("strong",[s("code",[e._v("nh1")])]),e._v("="),s("em",[s("code",[e._v("None")])]),e._v(", "),s("strong",[s("code",[e._v("nh2")])]),e._v("="),s("em",[s("code",[e._v("None")])]),e._v(", "),s("strong",[s("code",[e._v("dw")])]),e._v("="),s("em",[s("code",[e._v("False")])]),e._v(", "),s("strong",[s("code",[e._v("g2")])]),e._v("="),s("em",[s("code",[e._v("1")])]),e._v(", "),s("strong",[s("code",[e._v("sym")])]),e._v("="),s("em",[s("code",[e._v("False")])]),e._v(", "),s("strong",[s("code",[e._v("norm_type")])]),e._v("="),s("em",[s("code",[e._v("<NormType.Batch: 1>")])]),e._v(", "),s("strong",[s("code",[e._v("ndim")])]),e._v("="),s("em",[s("code",[e._v("2")])]),e._v(", "),s("strong",[s("code",[e._v("ks")])]),e._v("="),s("em",[s("code",[e._v("3")])]),e._v(", "),s("strong",[s("code",[e._v("pool")])]),e._v("="),s("em",[s("code",[e._v("AvgPool")])]),e._v(", "),s("strong",[s("code",[e._v("pool_first")])]),e._v("="),s("em",[s("code",[e._v("True")])]),e._v(", "),s("strong",[s("code",[e._v("padding")])]),e._v("="),s("em",[s("code",[e._v("None")])]),e._v(", "),s("strong",[s("code",[e._v("bias")])]),e._v("="),s("em",[s("code",[e._v("None")])]),e._v(", "),s("strong",[s("code",[e._v("bn_1st")])]),e._v("="),s("em",[s("code",[e._v("True")])]),e._v(", "),s("strong",[s("code",[e._v("transpose")])]),e._v("="),s("em",[s("code",[e._v("False")])]),e._v(", "),s("strong",[s("code",[e._v("init")])]),e._v("="),s("em",[s("code",[e._v("'auto'")])]),e._v(", "),s("strong",[s("code",[e._v("xtra")])]),e._v("="),s("em",[s("code",[e._v("None")])]),e._v(", "),s("strong",[s("code",[e._v("bias_std")])]),e._v("="),s("em",[s("code",[e._v("0.01")])]),e._v(", "),s("strong",[s("code",[e._v("dilation")])]),e._v(":"),s("code",[e._v("Union")]),e._v("["),s("code",[e._v("int")]),e._v(", "),s("code",[e._v("Tuple")]),e._v("["),s("code",[e._v("int")]),e._v(", "),s("code",[e._v("int")]),e._v("]]="),s("em",[s("code",[e._v("1")])]),e._v(", "),s("strong",[s("code",[e._v("padding_mode")])]),e._v(":"),s("code",[e._v("str")]),e._v("="),s("em",[s("code",[e._v("'zeros'")])]),e._v(") :: "),s("code",[e._v("Sequential")])])]),e._v(" "),s("p",[e._v("A sequential container.\nModules will be added to it in the order they are passed in the constructor.\nAlternatively, an ordered dict of modules can also be passed in.")]),e._v(" "),s("p",[e._v("To make it easier to understand, here is a small example::")]),e._v(" "),s("div",{staticClass:"language- extra-class"},[s("pre",[s("code",[e._v("# Example of using Sequential\nmodel = nn.Sequential(\n          nn.Conv2d(1,20,5),\n          nn.ReLU(),\n          nn.Conv2d(20,64,5),\n          nn.ReLU()\n        )\n\n# Example of using Sequential with OrderedDict\nmodel = nn.Sequential(OrderedDict([\n          ('conv1', nn.Conv2d(1,20,5)),\n          ('relu1', nn.ReLU()),\n          ('conv2', nn.Conv2d(20,64,5)),\n          ('relu2', nn.ReLU())\n        ]))\n")])])]),s("h4",{staticClass:"doc_header",attrs:{id:"xresnet18"}},[s("code",[e._v("xresnet18")]),s("a",{staticClass:"source_link",staticStyle:{float:"right"},attrs:{href:"https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L62"}},[e._v("[source]")])]),e._v(" "),s("blockquote",[s("p",[s("code",[e._v("xresnet18")]),e._v("("),s("strong",[s("code",[e._v("pretrained")])]),e._v("="),s("em",[s("code",[e._v("False")])]),e._v(", "),s("strong",[e._v("**"),s("code",[e._v("kwargs")])]),e._v(")")])]),e._v(" "),s("h4",{staticClass:"doc_header",attrs:{id:"xresnet34"}},[s("code",[e._v("xresnet34")]),s("a",{staticClass:"source_link",staticStyle:{float:"right"},attrs:{href:"https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L63"}},[e._v("[source]")])]),e._v(" "),s("blockquote",[s("p",[s("code",[e._v("xresnet34")]),e._v("("),s("strong",[s("code",[e._v("pretrained")])]),e._v("="),s("em",[s("code",[e._v("False")])]),e._v(", "),s("strong",[e._v("**"),s("code",[e._v("kwargs")])]),e._v(")")])]),e._v(" "),s("h4",{staticClass:"doc_header",attrs:{id:"xresnet50"}},[s("code",[e._v("xresnet50")]),s("a",{staticClass:"source_link",staticStyle:{float:"right"},attrs:{href:"https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L64"}},[e._v("[source]")])]),e._v(" "),s("blockquote",[s("p",[s("code",[e._v("xresnet50")]),e._v("("),s("strong",[s("code",[e._v("pretrained")])]),e._v("="),s("em",[s("code",[e._v("False")])]),e._v(", "),s("strong",[e._v("**"),s("code",[e._v("kwargs")])]),e._v(")")])]),e._v(" "),s("h4",{staticClass:"doc_header",attrs:{id:"xresnet101"}},[s("code",[e._v("xresnet101")]),s("a",{staticClass:"source_link",staticStyle:{float:"right"},attrs:{href:"https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L65"}},[e._v("[source]")])]),e._v(" "),s("blockquote",[s("p",[s("code",[e._v("xresnet101")]),e._v("("),s("strong",[s("code",[e._v("pretrained")])]),e._v("="),s("em",[s("code",[e._v("False")])]),e._v(", "),s("strong",[e._v("**"),s("code",[e._v("kwargs")])]),e._v(")")])]),e._v(" "),s("h4",{staticClass:"doc_header",attrs:{id:"xresnet152"}},[s("code",[e._v("xresnet152")]),s("a",{staticClass:"source_link",staticStyle:{float:"right"},attrs:{href:"https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L66"}},[e._v("[source]")])]),e._v(" "),s("blockquote",[s("p",[s("code",[e._v("xresnet152")]),e._v("("),s("strong",[s("code",[e._v("pretrained")])]),e._v("="),s("em",[s("code",[e._v("False")])]),e._v(", "),s("strong",[e._v("**"),s("code",[e._v("kwargs")])]),e._v(")")])]),e._v(" "),s("h4",{staticClass:"doc_header",attrs:{id:"xresnet18_deep"}},[s("code",[e._v("xresnet18_deep")]),s("a",{staticClass:"source_link",staticStyle:{float:"right"},attrs:{href:"https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L67"}},[e._v("[source]")])]),e._v(" "),s("blockquote",[s("p",[s("code",[e._v("xresnet18_deep")]),e._v("("),s("strong",[s("code",[e._v("pretrained")])]),e._v("="),s("em",[s("code",[e._v("False")])]),e._v(", "),s("strong",[e._v("**"),s("code",[e._v("kwargs")])]),e._v(")")])]),e._v(" "),s("h4",{staticClass:"doc_header",attrs:{id:"xresnet34_deep"}},[s("code",[e._v("xresnet34_deep")]),s("a",{staticClass:"source_link",staticStyle:{float:"right"},attrs:{href:"https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L68"}},[e._v("[source]")])]),e._v(" "),s("blockquote",[s("p",[s("code",[e._v("xresnet34_deep")]),e._v("("),s("strong",[s("code",[e._v("pretrained")])]),e._v("="),s("em",[s("code",[e._v("False")])]),e._v(", "),s("strong",[e._v("**"),s("code",[e._v("kwargs")])]),e._v(")")])]),e._v(" "),s("h4",{staticClass:"doc_header",attrs:{id:"xresnet50_deep"}},[s("code",[e._v("xresnet50_deep")]),s("a",{staticClass:"source_link",staticStyle:{float:"right"},attrs:{href:"https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L69"}},[e._v("[source]")])]),e._v(" "),s("blockquote",[s("p",[s("code",[e._v("xresnet50_deep")]),e._v("("),s("strong",[s("code",[e._v("pretrained")])]),e._v("="),s("em",[s("code",[e._v("False")])]),e._v(", "),s("strong",[e._v("**"),s("code",[e._v("kwargs")])]),e._v(")")])]),e._v(" "),s("h4",{staticClass:"doc_header",attrs:{id:"xresnet18_deeper"}},[s("code",[e._v("xresnet18_deeper")]),s("a",{staticClass:"source_link",staticStyle:{float:"right"},attrs:{href:"https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L70"}},[e._v("[source]")])]),e._v(" "),s("blockquote",[s("p",[s("code",[e._v("xresnet18_deeper")]),e._v("("),s("strong",[s("code",[e._v("pretrained")])]),e._v("="),s("em",[s("code",[e._v("False")])]),e._v(", "),s("strong",[e._v("**"),s("code",[e._v("kwargs")])]),e._v(")")])]),e._v(" "),s("h4",{staticClass:"doc_header",attrs:{id:"xresnet34_deeper"}},[s("code",[e._v("xresnet34_deeper")]),s("a",{staticClass:"source_link",staticStyle:{float:"right"},attrs:{href:"https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L71"}},[e._v("[source]")])]),e._v(" "),s("blockquote",[s("p",[s("code",[e._v("xresnet34_deeper")]),e._v("("),s("strong",[s("code",[e._v("pretrained")])]),e._v("="),s("em",[s("code",[e._v("False")])]),e._v(", "),s("strong",[e._v("**"),s("code",[e._v("kwargs")])]),e._v(")")])]),e._v(" "),s("h4",{staticClass:"doc_header",attrs:{id:"xresnet50_deeper"}},[s("code",[e._v("xresnet50_deeper")]),s("a",{staticClass:"source_link",staticStyle:{float:"right"},attrs:{href:"https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L72"}},[e._v("[source]")])]),e._v(" "),s("blockquote",[s("p",[s("code",[e._v("xresnet50_deeper")]),e._v("("),s("strong",[s("code",[e._v("pretrained")])]),e._v("="),s("em",[s("code",[e._v("False")])]),e._v(", "),s("strong",[e._v("**"),s("code",[e._v("kwargs")])]),e._v(")")])]),e._v(" "),s("h4",{staticClass:"doc_header",attrs:{id:"xse_resnet18"}},[s("code",[e._v("xse_resnet18")]),s("a",{staticClass:"source_link",staticStyle:{float:"right"},attrs:{href:"https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L84"}},[e._v("[source]")])]),e._v(" "),s("blockquote",[s("p",[s("code",[e._v("xse_resnet18")]),e._v("("),s("strong",[s("code",[e._v("n_out")])]),e._v("="),s("em",[s("code",[e._v("1000")])]),e._v(", "),s("strong",[s("code",[e._v("pretrained")])]),e._v("="),s("em",[s("code",[e._v("False")])]),e._v(", "),s("strong",[e._v("**"),s("code",[e._v("kwargs")])]),e._v(")")])]),e._v(" "),s("h4",{staticClass:"doc_header",attrs:{id:"xse_resnext18"}},[s("code",[e._v("xse_resnext18")]),s("a",{staticClass:"source_link",staticStyle:{float:"right"},attrs:{href:"https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L85"}},[e._v("[source]")])]),e._v(" "),s("blockquote",[s("p",[s("code",[e._v("xse_resnext18")]),e._v("("),s("strong",[s("code",[e._v("n_out")])]),e._v("="),s("em",[s("code",[e._v("1000")])]),e._v(", "),s("strong",[s("code",[e._v("pretrained")])]),e._v("="),s("em",[s("code",[e._v("False")])]),e._v(", "),s("strong",[e._v("**"),s("code",[e._v("kwargs")])]),e._v(")")])]),e._v(" "),s("h4",{staticClass:"doc_header",attrs:{id:"xresnext18"}},[s("code",[e._v("xresnext18")]),s("a",{staticClass:"source_link",staticStyle:{float:"right"},attrs:{href:"https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L86"}},[e._v("[source]")])]),e._v(" "),s("blockquote",[s("p",[s("code",[e._v("xresnext18")]),e._v("("),s("strong",[s("code",[e._v("n_out")])]),e._v("="),s("em",[s("code",[e._v("1000")])]),e._v(", "),s("strong",[s("code",[e._v("pretrained")])]),e._v("="),s("em",[s("code",[e._v("False")])]),e._v(", "),s("strong",[e._v("**"),s("code",[e._v("kwargs")])]),e._v(")")])]),e._v(" "),s("h4",{staticClass:"doc_header",attrs:{id:"xse_resnet34"}},[s("code",[e._v("xse_resnet34")]),s("a",{staticClass:"source_link",staticStyle:{float:"right"},attrs:{href:"https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L87"}},[e._v("[source]")])]),e._v(" "),s("blockquote",[s("p",[s("code",[e._v("xse_resnet34")]),e._v("("),s("strong",[s("code",[e._v("n_out")])]),e._v("="),s("em",[s("code",[e._v("1000")])]),e._v(", "),s("strong",[s("code",[e._v("pretrained")])]),e._v("="),s("em",[s("code",[e._v("False")])]),e._v(", "),s("strong",[e._v("**"),s("code",[e._v("kwargs")])]),e._v(")")])]),e._v(" "),s("h4",{staticClass:"doc_header",attrs:{id:"xse_resnext34"}},[s("code",[e._v("xse_resnext34")]),s("a",{staticClass:"source_link",staticStyle:{float:"right"},attrs:{href:"https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L88"}},[e._v("[source]")])]),e._v(" "),s("blockquote",[s("p",[s("code",[e._v("xse_resnext34")]),e._v("("),s("strong",[s("code",[e._v("n_out")])]),e._v("="),s("em",[s("code",[e._v("1000")])]),e._v(", "),s("strong",[s("code",[e._v("pretrained")])]),e._v("="),s("em",[s("code",[e._v("False")])]),e._v(", "),s("strong",[e._v("**"),s("code",[e._v("kwargs")])]),e._v(")")])]),e._v(" "),s("h4",{staticClass:"doc_header",attrs:{id:"xresnext34"}},[s("code",[e._v("xresnext34")]),s("a",{staticClass:"source_link",staticStyle:{float:"right"},attrs:{href:"https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L89"}},[e._v("[source]")])]),e._v(" "),s("blockquote",[s("p",[s("code",[e._v("xresnext34")]),e._v("("),s("strong",[s("code",[e._v("n_out")])]),e._v("="),s("em",[s("code",[e._v("1000")])]),e._v(", "),s("strong",[s("code",[e._v("pretrained")])]),e._v("="),s("em",[s("code",[e._v("False")])]),e._v(", "),s("strong",[e._v("**"),s("code",[e._v("kwargs")])]),e._v(")")])]),e._v(" "),s("h4",{staticClass:"doc_header",attrs:{id:"xse_resnet50"}},[s("code",[e._v("xse_resnet50")]),s("a",{staticClass:"source_link",staticStyle:{float:"right"},attrs:{href:"https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L90"}},[e._v("[source]")])]),e._v(" "),s("blockquote",[s("p",[s("code",[e._v("xse_resnet50")]),e._v("("),s("strong",[s("code",[e._v("n_out")])]),e._v("="),s("em",[s("code",[e._v("1000")])]),e._v(", "),s("strong",[s("code",[e._v("pretrained")])]),e._v("="),s("em",[s("code",[e._v("False")])]),e._v(", "),s("strong",[e._v("**"),s("code",[e._v("kwargs")])]),e._v(")")])]),e._v(" "),s("h4",{staticClass:"doc_header",attrs:{id:"xse_resnext50"}},[s("code",[e._v("xse_resnext50")]),s("a",{staticClass:"source_link",staticStyle:{float:"right"},attrs:{href:"https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L91"}},[e._v("[source]")])]),e._v(" "),s("blockquote",[s("p",[s("code",[e._v("xse_resnext50")]),e._v("("),s("strong",[s("code",[e._v("n_out")])]),e._v("="),s("em",[s("code",[e._v("1000")])]),e._v(", "),s("strong",[s("code",[e._v("pretrained")])]),e._v("="),s("em",[s("code",[e._v("False")])]),e._v(", "),s("strong",[e._v("**"),s("code",[e._v("kwargs")])]),e._v(")")])]),e._v(" "),s("h4",{staticClass:"doc_header",attrs:{id:"xresnext50"}},[s("code",[e._v("xresnext50")]),s("a",{staticClass:"source_link",staticStyle:{float:"right"},attrs:{href:"https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L92"}},[e._v("[source]")])]),e._v(" "),s("blockquote",[s("p",[s("code",[e._v("xresnext50")]),e._v("("),s("strong",[s("code",[e._v("n_out")])]),e._v("="),s("em",[s("code",[e._v("1000")])]),e._v(", "),s("strong",[s("code",[e._v("pretrained")])]),e._v("="),s("em",[s("code",[e._v("False")])]),e._v(", "),s("strong",[e._v("**"),s("code",[e._v("kwargs")])]),e._v(")")])]),e._v(" "),s("h4",{staticClass:"doc_header",attrs:{id:"xse_resnet101"}},[s("code",[e._v("xse_resnet101")]),s("a",{staticClass:"source_link",staticStyle:{float:"right"},attrs:{href:"https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L93"}},[e._v("[source]")])]),e._v(" "),s("blockquote",[s("p",[s("code",[e._v("xse_resnet101")]),e._v("("),s("strong",[s("code",[e._v("n_out")])]),e._v("="),s("em",[s("code",[e._v("1000")])]),e._v(", "),s("strong",[s("code",[e._v("pretrained")])]),e._v("="),s("em",[s("code",[e._v("False")])]),e._v(", "),s("strong",[e._v("**"),s("code",[e._v("kwargs")])]),e._v(")")])]),e._v(" "),s("h4",{staticClass:"doc_header",attrs:{id:"xse_resnext101"}},[s("code",[e._v("xse_resnext101")]),s("a",{staticClass:"source_link",staticStyle:{float:"right"},attrs:{href:"https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L94"}},[e._v("[source]")])]),e._v(" "),s("blockquote",[s("p",[s("code",[e._v("xse_resnext101")]),e._v("("),s("strong",[s("code",[e._v("n_out")])]),e._v("="),s("em",[s("code",[e._v("1000")])]),e._v(", "),s("strong",[s("code",[e._v("pretrained")])]),e._v("="),s("em",[s("code",[e._v("False")])]),e._v(", "),s("strong",[e._v("**"),s("code",[e._v("kwargs")])]),e._v(")")])]),e._v(" "),s("h4",{staticClass:"doc_header",attrs:{id:"xresnext101"}},[s("code",[e._v("xresnext101")]),s("a",{staticClass:"source_link",staticStyle:{float:"right"},attrs:{href:"https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L95"}},[e._v("[source]")])]),e._v(" "),s("blockquote",[s("p",[s("code",[e._v("xresnext101")]),e._v("("),s("strong",[s("code",[e._v("n_out")])]),e._v("="),s("em",[s("code",[e._v("1000")])]),e._v(", "),s("strong",[s("code",[e._v("pretrained")])]),e._v("="),s("em",[s("code",[e._v("False")])]),e._v(", "),s("strong",[e._v("**"),s("code",[e._v("kwargs")])]),e._v(")")])]),e._v(" "),s("h4",{staticClass:"doc_header",attrs:{id:"xse_resnet152"}},[s("code",[e._v("xse_resnet152")]),s("a",{staticClass:"source_link",staticStyle:{float:"right"},attrs:{href:"https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L96"}},[e._v("[source]")])]),e._v(" "),s("blockquote",[s("p",[s("code",[e._v("xse_resnet152")]),e._v("("),s("strong",[s("code",[e._v("n_out")])]),e._v("="),s("em",[s("code",[e._v("1000")])]),e._v(", "),s("strong",[s("code",[e._v("pretrained")])]),e._v("="),s("em",[s("code",[e._v("False")])]),e._v(", "),s("strong",[e._v("**"),s("code",[e._v("kwargs")])]),e._v(")")])]),e._v(" "),s("h4",{staticClass:"doc_header",attrs:{id:"xsenet154"}},[s("code",[e._v("xsenet154")]),s("a",{staticClass:"source_link",staticStyle:{float:"right"},attrs:{href:"https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L97"}},[e._v("[source]")])]),e._v(" "),s("blockquote",[s("p",[s("code",[e._v("xsenet154")]),e._v("("),s("strong",[s("code",[e._v("n_out")])]),e._v("="),s("em",[s("code",[e._v("1000")])]),e._v(", "),s("strong",[s("code",[e._v("pretrained")])]),e._v("="),s("em",[s("code",[e._v("False")])]),e._v(", "),s("strong",[e._v("**"),s("code",[e._v("kwargs")])]),e._v(")")])]),e._v(" "),s("h4",{staticClass:"doc_header",attrs:{id:"xse_resnext18_deep"}},[s("code",[e._v("xse_resnext18_deep")]),s("a",{staticClass:"source_link",staticStyle:{float:"right"},attrs:{href:"https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L99"}},[e._v("[source]")])]),e._v(" "),s("blockquote",[s("p",[s("code",[e._v("xse_resnext18_deep")]),e._v("("),s("strong",[s("code",[e._v("n_out")])]),e._v("="),s("em",[s("code",[e._v("1000")])]),e._v(", "),s("strong",[s("code",[e._v("pretrained")])]),e._v("="),s("em",[s("code",[e._v("False")])]),e._v(", "),s("strong",[e._v("**"),s("code",[e._v("kwargs")])]),e._v(")")])]),e._v(" "),s("h4",{staticClass:"doc_header",attrs:{id:"xse_resnext34_deep"}},[s("code",[e._v("xse_resnext34_deep")]),s("a",{staticClass:"source_link",staticStyle:{float:"right"},attrs:{href:"https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L100"}},[e._v("[source]")])]),e._v(" "),s("blockquote",[s("p",[s("code",[e._v("xse_resnext34_deep")]),e._v("("),s("strong",[s("code",[e._v("n_out")])]),e._v("="),s("em",[s("code",[e._v("1000")])]),e._v(", "),s("strong",[s("code",[e._v("pretrained")])]),e._v("="),s("em",[s("code",[e._v("False")])]),e._v(", "),s("strong",[e._v("**"),s("code",[e._v("kwargs")])]),e._v(")")])]),e._v(" "),s("h4",{staticClass:"doc_header",attrs:{id:"xse_resnext50_deep"}},[s("code",[e._v("xse_resnext50_deep")]),s("a",{staticClass:"source_link",staticStyle:{float:"right"},attrs:{href:"https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L101"}},[e._v("[source]")])]),e._v(" "),s("blockquote",[s("p",[s("code",[e._v("xse_resnext50_deep")]),e._v("("),s("strong",[s("code",[e._v("n_out")])]),e._v("="),s("em",[s("code",[e._v("1000")])]),e._v(", "),s("strong",[s("code",[e._v("pretrained")])]),e._v("="),s("em",[s("code",[e._v("False")])]),e._v(", "),s("strong",[e._v("**"),s("code",[e._v("kwargs")])]),e._v(")")])]),e._v(" "),s("h4",{staticClass:"doc_header",attrs:{id:"xse_resnext18_deeper"}},[s("code",[e._v("xse_resnext18_deeper")]),s("a",{staticClass:"source_link",staticStyle:{float:"right"},attrs:{href:"https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L102"}},[e._v("[source]")])]),e._v(" "),s("blockquote",[s("p",[s("code",[e._v("xse_resnext18_deeper")]),e._v("("),s("strong",[s("code",[e._v("n_out")])]),e._v("="),s("em",[s("code",[e._v("1000")])]),e._v(", "),s("strong",[s("code",[e._v("pretrained")])]),e._v("="),s("em",[s("code",[e._v("False")])]),e._v(", "),s("strong",[e._v("**"),s("code",[e._v("kwargs")])]),e._v(")")])]),e._v(" "),s("h4",{staticClass:"doc_header",attrs:{id:"xse_resnext34_deeper"}},[s("code",[e._v("xse_resnext34_deeper")]),s("a",{staticClass:"source_link",staticStyle:{float:"right"},attrs:{href:"https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L103"}},[e._v("[source]")])]),e._v(" "),s("blockquote",[s("p",[s("code",[e._v("xse_resnext34_deeper")]),e._v("("),s("strong",[s("code",[e._v("n_out")])]),e._v("="),s("em",[s("code",[e._v("1000")])]),e._v(", "),s("strong",[s("code",[e._v("pretrained")])]),e._v("="),s("em",[s("code",[e._v("False")])]),e._v(", "),s("strong",[e._v("**"),s("code",[e._v("kwargs")])]),e._v(")")])]),e._v(" "),s("h4",{staticClass:"doc_header",attrs:{id:"xse_resnext50_deeper"}},[s("code",[e._v("xse_resnext50_deeper")]),s("a",{staticClass:"source_link",staticStyle:{float:"right"},attrs:{href:"https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L104"}},[e._v("[source]")])]),e._v(" "),s("blockquote",[s("p",[s("code",[e._v("xse_resnext50_deeper")]),e._v("("),s("strong",[s("code",[e._v("n_out")])]),e._v("="),s("em",[s("code",[e._v("1000")])]),e._v(", "),s("strong",[s("code",[e._v("pretrained")])]),e._v("="),s("em",[s("code",[e._v("False")])]),e._v(", "),s("strong",[e._v("**"),s("code",[e._v("kwargs")])]),e._v(")")])]),e._v(" "),s("div",{staticClass:"language-python extra-class"},[s("pre",{pre:!0,attrs:{class:"language-python"}},[s("code",[e._v("tst "),s("span",{pre:!0,attrs:{class:"token operator"}},[e._v("=")]),e._v(" xse_resnext18"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v("(")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v(")")]),e._v("\nx "),s("span",{pre:!0,attrs:{class:"token operator"}},[e._v("=")]),e._v(" torch"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v(".")]),e._v("randn"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v("(")]),s("span",{pre:!0,attrs:{class:"token number"}},[e._v("64")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v(",")]),e._v(" "),s("span",{pre:!0,attrs:{class:"token number"}},[e._v("3")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v(",")]),e._v(" "),s("span",{pre:!0,attrs:{class:"token number"}},[e._v("128")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v(",")]),e._v(" "),s("span",{pre:!0,attrs:{class:"token number"}},[e._v("128")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v(")")]),e._v("\ny "),s("span",{pre:!0,attrs:{class:"token operator"}},[e._v("=")]),e._v(" tst"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v("(")]),e._v("x"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v(")")]),e._v("\n")])])]),s("div",{staticClass:"language-python extra-class"},[s("pre",{pre:!0,attrs:{class:"language-python"}},[s("code",[e._v("tst "),s("span",{pre:!0,attrs:{class:"token operator"}},[e._v("=")]),e._v(" xresnext18"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v("(")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v(")")]),e._v("\nx "),s("span",{pre:!0,attrs:{class:"token operator"}},[e._v("=")]),e._v(" torch"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v(".")]),e._v("randn"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v("(")]),s("span",{pre:!0,attrs:{class:"token number"}},[e._v("64")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v(",")]),e._v(" "),s("span",{pre:!0,attrs:{class:"token number"}},[e._v("3")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v(",")]),e._v(" "),s("span",{pre:!0,attrs:{class:"token number"}},[e._v("128")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v(",")]),e._v(" "),s("span",{pre:!0,attrs:{class:"token number"}},[e._v("128")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v(")")]),e._v("\ny "),s("span",{pre:!0,attrs:{class:"token operator"}},[e._v("=")]),e._v(" tst"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v("(")]),e._v("x"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v(")")]),e._v("\n")])])]),s("div",{staticClass:"language-python extra-class"},[s("pre",{pre:!0,attrs:{class:"language-python"}},[s("code",[e._v("tst "),s("span",{pre:!0,attrs:{class:"token operator"}},[e._v("=")]),e._v(" xse_resnet50"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v("(")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v(")")]),e._v("\nx "),s("span",{pre:!0,attrs:{class:"token operator"}},[e._v("=")]),e._v(" torch"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v(".")]),e._v("randn"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v("(")]),s("span",{pre:!0,attrs:{class:"token number"}},[e._v("8")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v(",")]),e._v(" "),s("span",{pre:!0,attrs:{class:"token number"}},[e._v("3")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v(",")]),e._v(" "),s("span",{pre:!0,attrs:{class:"token number"}},[e._v("64")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v(",")]),e._v(" "),s("span",{pre:!0,attrs:{class:"token number"}},[e._v("64")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v(")")]),e._v("\ny "),s("span",{pre:!0,attrs:{class:"token operator"}},[e._v("=")]),e._v(" tst"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v("(")]),e._v("x"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v(")")]),e._v("\n")])])])])}),[],!1,null,null,null);t.default=_.exports}}]);