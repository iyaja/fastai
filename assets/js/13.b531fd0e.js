(window.webpackJsonp=window.webpackJsonp||[]).push([[13],{405:function(e,t,a){"use strict";a.r(t);var n=a(42),r=Object(n.a)({},(function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("ContentSlotsDistributor",{attrs:{"slot-key":e.$parent.slotKey}},[a("h1",{attrs:{id:"neptune-ai"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#neptune-ai"}},[e._v("#")]),e._v(" Neptune.ai")]),e._v(" "),a("blockquote",[a("p",[e._v("Integration with "),a("a",{attrs:{href:"https://www.neptune.ai"}},[e._v("neptune.ai")]),e._v(".")])]),e._v(" "),a("h2",{attrs:{id:"registration"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#registration"}},[e._v("#")]),e._v(" Registration")]),e._v(" "),a("ol",[a("li",[e._v("Create "),a("strong",[e._v("free")]),e._v(" account: "),a("a",{attrs:{href:"https://neptune.ai/register",target:"_blank",rel:"noopener noreferrer"}},[e._v("neptune.ai/register"),a("OutboundLink")],1),e._v(".")]),e._v(" "),a("li",[e._v("Export API token to the environment variable (more help "),a("a",{attrs:{href:"https://docs.neptune.ai/python-api/tutorials/get-started.html#copy-api-token",target:"_blank",rel:"noopener noreferrer"}},[e._v("here"),a("OutboundLink")],1),e._v("). In your terminal run:")])]),e._v(" "),a("div",{staticClass:"language- extra-class"},[a("pre",{pre:!0,attrs:{class:"language-text"}},[a("code",[e._v("export NEPTUNE_API_TOKEN='YOUR_LONG_API_TOKEN'\n")])])]),a("p",[e._v("or append the command above to your "),a("code",[e._v("~/.bashrc")]),e._v(" or "),a("code",[e._v("~/.bash_profile")]),e._v(" files ("),a("strong",[e._v("recommended")]),e._v("). More help is "),a("a",{attrs:{href:"https://docs.neptune.ai/python-api/tutorials/get-started.html#copy-api-token",target:"_blank",rel:"noopener noreferrer"}},[e._v("here"),a("OutboundLink")],1),e._v(".")]),e._v(" "),a("h2",{attrs:{id:"installation"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#installation"}},[e._v("#")]),e._v(" Installation")]),e._v(" "),a("ol",[a("li",[e._v("You need to install neptune-client. In your terminal run:")])]),e._v(" "),a("div",{staticClass:"language- extra-class"},[a("pre",{pre:!0,attrs:{class:"language-text"}},[a("code",[e._v("pip install neptune-client\n")])])]),a("p",[e._v("or (alternative installation using conda). In your terminal run:")]),e._v(" "),a("div",{staticClass:"language- extra-class"},[a("pre",{pre:!0,attrs:{class:"language-text"}},[a("code",[e._v("conda install neptune-client -c conda-forge\n")])])]),a("ol",{attrs:{start:"2"}},[a("li",[e._v("Install "),a("a",{attrs:{href:"https://psutil.readthedocs.io/en/latest/",target:"_blank",rel:"noopener noreferrer"}},[e._v("psutil"),a("OutboundLink")],1),e._v(" to see hardware monitoring charts:")])]),e._v(" "),a("div",{staticClass:"language- extra-class"},[a("pre",{pre:!0,attrs:{class:"language-text"}},[a("code",[e._v("pip install psutil\n")])])]),a("h2",{attrs:{id:"how-to-use"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#how-to-use"}},[e._v("#")]),e._v(" How to use?")]),e._v(" "),a("p",[e._v("Key is to call "),a("code",[e._v("neptune.init()")]),e._v(" before you create "),a("code",[e._v("Learner()")]),e._v(" and call "),a("code",[e._v("neptune_create_experiment()")]),e._v(", before you fit the model.")]),e._v(" "),a("p",[e._v("Use "),a("RouterLink",{attrs:{to:"/callback.neptune.html#NeptuneCallback"}},[a("code",[e._v("NeptuneCallback")])]),e._v(" in your "),a("RouterLink",{attrs:{to:"/learner.html#Learner"}},[a("code",[e._v("Learner")])]),e._v(", like this:")],1),e._v(" "),a("div",{staticClass:"language- extra-class"},[a("pre",{pre:!0,attrs:{class:"language-text"}},[a("code",[e._v("from fastai.callback.neptune import NeptuneCallback\n\nneptune.init('USERNAME/PROJECT_NAME')  # specify project\n\nlearn = Learner(dls, model,\n                cbs=NeptuneCallback()\n                )\n\nneptune.create_experiment()  # start experiment\nlearn.fit_one_cycle(1)\n")])])]),a("h2",{staticClass:"doc_header",attrs:{id:"NeptuneCallback"}},[a("code",[e._v("class")]),e._v(" "),a("code",[e._v("NeptuneCallback")]),a("a",{staticClass:"source_link",staticStyle:{float:"right"},attrs:{href:"https://github.com/fastai/fastai/tree/master/fastai/callback/neptune.py#L14"}},[e._v("[source]")])]),e._v(" "),a("blockquote",[a("p",[a("code",[e._v("NeptuneCallback")]),e._v("("),a("strong",[a("code",[e._v("log_model_weights")])]),e._v("="),a("em",[a("code",[e._v("True")])]),e._v(", "),a("strong",[a("code",[e._v("keep_experiment_running")])]),e._v("="),a("em",[a("code",[e._v("False")])]),e._v(") :: "),a("RouterLink",{attrs:{to:"/callback.core.html#Callback"}},[a("code",[e._v("Callback")])])],1)]),e._v(" "),a("p",[e._v("Log losses, metrics, model weights, model architecture summary to neptune")])])}),[],!1,null,null,null);t.default=r.exports}}]);