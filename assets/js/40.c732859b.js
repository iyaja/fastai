(window.webpackJsonp=window.webpackJsonp||[]).push([[40],{404:function(t,a,e){"use strict";e.r(a);var s=e(42),r=Object(s.a)({},(function(){var t=this,a=t.$createElement,e=t._self._c||a;return e("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[e("h1",{attrs:{id:"tabular-learner"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#tabular-learner"}},[t._v("#")]),t._v(" Tabular learner")]),t._v(" "),e("blockquote",[e("p",[t._v("The function to immediately get a "),e("code",[t._v("Learner")]),t._v(" ready to train for tabular data")])]),t._v(" "),e("div",{staticClass:"language-python extra-class"},[e("pre",{pre:!0,attrs:{class:"language-python"}},[e("code",[e("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("from")]),t._v(" fastai"),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("tabular"),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("data "),e("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("import")]),t._v(" "),e("span",{pre:!0,attrs:{class:"token operator"}},[t._v("*")]),t._v("\n")])])]),e("p",[t._v("The main function you probably want to use in this module is "),e("RouterLink",{attrs:{to:"/tabular.learner.html#tabular_learner"}},[e("code",[t._v("tabular_learner")])]),t._v(". It will automatically create a "),e("RouterLink",{attrs:{to:"/tabular.model.html#TabularModel"}},[e("code",[t._v("TabularModel")])]),t._v(" suitable for your data and infer the right loss function. See the "),e("a",{attrs:{href:"http://docs.fast.ai/tutorial.tabular",target:"_blank",rel:"noopener noreferrer"}},[t._v("tabular tutorial"),e("OutboundLink")],1),t._v(" for an example of use in context.")],1),t._v(" "),e("h2",{attrs:{id:"main-functions"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#main-functions"}},[t._v("#")]),t._v(" Main functions")]),t._v(" "),e("h3",{staticClass:"doc_header",attrs:{id:"TabularLearner"}},[e("code",[t._v("class")]),t._v(" "),e("code",[t._v("TabularLearner")]),e("a",{staticClass:"source_link",staticStyle:{float:"right"},attrs:{href:"https://github.com/fastai/fastai/tree/master/fastai/tabular/learner.py#L12"}},[t._v("[source]")])]),t._v(" "),e("blockquote",[e("p",[e("code",[t._v("TabularLearner")]),t._v("("),e("strong",[e("code",[t._v("dls")])]),t._v(", "),e("strong",[e("code",[t._v("model")])]),t._v(", "),e("strong",[e("code",[t._v("loss_func")])]),t._v("="),e("em",[e("code",[t._v("None")])]),t._v(", "),e("strong",[e("code",[t._v("opt_func")])]),t._v("="),e("em",[e("code",[t._v("Adam")])]),t._v(", "),e("strong",[e("code",[t._v("lr")])]),t._v("="),e("em",[e("code",[t._v("0.001")])]),t._v(", "),e("strong",[e("code",[t._v("splitter")])]),t._v("="),e("em",[e("code",[t._v("trainable_params")])]),t._v(", "),e("strong",[e("code",[t._v("cbs")])]),t._v("="),e("em",[e("code",[t._v("None")])]),t._v(", "),e("strong",[e("code",[t._v("metrics")])]),t._v("="),e("em",[e("code",[t._v("None")])]),t._v(", "),e("strong",[e("code",[t._v("path")])]),t._v("="),e("em",[e("code",[t._v("None")])]),t._v(", "),e("strong",[e("code",[t._v("model_dir")])]),t._v("="),e("em",[e("code",[t._v("'models'")])]),t._v(", "),e("strong",[e("code",[t._v("wd")])]),t._v("="),e("em",[e("code",[t._v("None")])]),t._v(", "),e("strong",[e("code",[t._v("wd_bn_bias")])]),t._v("="),e("em",[e("code",[t._v("False")])]),t._v(", "),e("strong",[e("code",[t._v("train_bn")])]),t._v("="),e("em",[e("code",[t._v("True")])]),t._v(", "),e("strong",[e("code",[t._v("moms")])]),t._v("="),e("em",[e("code",[t._v("(0.95, 0.85, 0.95)")])]),t._v(") :: "),e("RouterLink",{attrs:{to:"/learner.html#Learner"}},[e("code",[t._v("Learner")])])],1)]),t._v(" "),e("p",[e("RouterLink",{attrs:{to:"/learner.html#Learner"}},[e("code",[t._v("Learner")])]),t._v(" for tabular data")],1),t._v(" "),e("p",[t._v("It works exactly as a normal "),e("RouterLink",{attrs:{to:"/learner.html#Learner"}},[e("code",[t._v("Learner")])]),t._v(", the only difference is that it implements a "),e("code",[t._v("predict")]),t._v(" method specific to work on a row of data.")],1),t._v(" "),e("h4",{staticClass:"doc_header",attrs:{id:"tabular_learner"}},[e("code",[t._v("tabular_learner")]),e("a",{staticClass:"source_link",staticStyle:{float:"right"},attrs:{href:"https://github.com/fastai/fastai/tree/master/fastai/tabular/learner.py#L26"}},[t._v("[source]")])]),t._v(" "),e("blockquote",[e("p",[e("code",[t._v("tabular_learner")]),t._v("("),e("strong",[e("code",[t._v("dls")])]),t._v(", "),e("strong",[e("code",[t._v("layers")])]),t._v("="),e("em",[e("code",[t._v("None")])]),t._v(", "),e("strong",[e("code",[t._v("emb_szs")])]),t._v("="),e("em",[e("code",[t._v("None")])]),t._v(", "),e("strong",[e("code",[t._v("config")])]),t._v("="),e("em",[e("code",[t._v("None")])]),t._v(", "),e("strong",[e("code",[t._v("n_out")])]),t._v("="),e("em",[e("code",[t._v("None")])]),t._v(", "),e("strong",[e("code",[t._v("y_range")])]),t._v("="),e("em",[e("code",[t._v("None")])]),t._v(", "),e("strong",[e("code",[t._v("loss_func")])]),t._v("="),e("em",[e("code",[t._v("None")])]),t._v(", "),e("strong",[e("code",[t._v("opt_func")])]),t._v("="),e("em",[e("code",[t._v("Adam")])]),t._v(", "),e("strong",[e("code",[t._v("lr")])]),t._v("="),e("em",[e("code",[t._v("0.001")])]),t._v(", "),e("strong",[e("code",[t._v("splitter")])]),t._v("="),e("em",[e("code",[t._v("trainable_params")])]),t._v(", "),e("strong",[e("code",[t._v("cbs")])]),t._v("="),e("em",[e("code",[t._v("None")])]),t._v(", "),e("strong",[e("code",[t._v("metrics")])]),t._v("="),e("em",[e("code",[t._v("None")])]),t._v(", "),e("strong",[e("code",[t._v("path")])]),t._v("="),e("em",[e("code",[t._v("None")])]),t._v(", "),e("strong",[e("code",[t._v("model_dir")])]),t._v("="),e("em",[e("code",[t._v("'models'")])]),t._v(", "),e("strong",[e("code",[t._v("wd")])]),t._v("="),e("em",[e("code",[t._v("None")])]),t._v(", "),e("strong",[e("code",[t._v("wd_bn_bias")])]),t._v("="),e("em",[e("code",[t._v("False")])]),t._v(", "),e("strong",[e("code",[t._v("train_bn")])]),t._v("="),e("em",[e("code",[t._v("True")])]),t._v(", "),e("strong",[e("code",[t._v("moms")])]),t._v("="),e("em",[e("code",[t._v("(0.95, 0.85, 0.95)")])]),t._v(")")])]),t._v(" "),e("p",[t._v("Get a "),e("RouterLink",{attrs:{to:"/learner.html#Learner"}},[e("code",[t._v("Learner")])]),t._v(" using "),e("code",[t._v("dls")]),t._v(", with "),e("code",[t._v("metrics")]),t._v(", including a "),e("RouterLink",{attrs:{to:"/tabular.model.html#TabularModel"}},[e("code",[t._v("TabularModel")])]),t._v(" created using the remaining params.")],1),t._v(" "),e("p",[t._v("If your data was built with fastai, you probably won't need to pass anything to "),e("code",[t._v("emb_szs")]),t._v(" unless you want to change the default of the library (produced by "),e("RouterLink",{attrs:{to:"/tabular.model.html#get_emb_sz"}},[e("code",[t._v("get_emb_sz")])]),t._v("), same for "),e("code",[t._v("n_out")]),t._v(" which should be automatically inferred. "),e("RouterLink",{attrs:{to:"/layers.html"}},[e("code",[t._v("layers")])]),t._v(" will default to "),e("code",[t._v("[200,100]")]),t._v(" and is passed to "),e("RouterLink",{attrs:{to:"/tabular.model.html#TabularModel"}},[e("code",[t._v("TabularModel")])]),t._v(" along with the "),e("code",[t._v("config")]),t._v(".")],1),t._v(" "),e("p",[t._v("Use "),e("RouterLink",{attrs:{to:"/tabular.model.html#tabular_config"}},[e("code",[t._v("tabular_config")])]),t._v(" to create a "),e("code",[t._v("config")]),t._v(" and customize the model used. There is just easy access to "),e("code",[t._v("y_range")]),t._v(" because this argument is often used.")],1),t._v(" "),e("p",[t._v("All the other arguments are passed to "),e("RouterLink",{attrs:{to:"/learner.html#Learner"}},[e("code",[t._v("Learner")])]),t._v(".")],1),t._v(" "),e("div",{staticClass:"language-python extra-class"},[e("pre",{pre:!0,attrs:{class:"language-python"}},[e("code",[t._v("path "),e("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" untar_data"),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("URLs"),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("ADULT_SAMPLE"),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\ndf "),e("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" pd"),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("read_csv"),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("path"),e("span",{pre:!0,attrs:{class:"token operator"}},[t._v("/")]),e("span",{pre:!0,attrs:{class:"token string"}},[t._v("'adult.csv'")]),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\ncat_names "),e("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" "),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("[")]),e("span",{pre:!0,attrs:{class:"token string"}},[t._v("'workclass'")]),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" "),e("span",{pre:!0,attrs:{class:"token string"}},[t._v("'education'")]),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" "),e("span",{pre:!0,attrs:{class:"token string"}},[t._v("'marital-status'")]),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" "),e("span",{pre:!0,attrs:{class:"token string"}},[t._v("'occupation'")]),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" "),e("span",{pre:!0,attrs:{class:"token string"}},[t._v("'relationship'")]),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" "),e("span",{pre:!0,attrs:{class:"token string"}},[t._v("'race'")]),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("]")]),t._v("\ncont_names "),e("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" "),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("[")]),e("span",{pre:!0,attrs:{class:"token string"}},[t._v("'age'")]),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" "),e("span",{pre:!0,attrs:{class:"token string"}},[t._v("'fnlwgt'")]),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" "),e("span",{pre:!0,attrs:{class:"token string"}},[t._v("'education-num'")]),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("]")]),t._v("\nprocs "),e("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" "),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("[")]),t._v("Categorify"),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" FillMissing"),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" Normalize"),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("]")]),t._v("\ndls "),e("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" TabularDataLoaders"),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("from_df"),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("df"),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" path"),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" procs"),e("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v("procs"),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" cat_names"),e("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v("cat_names"),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" cont_names"),e("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v("cont_names"),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" \n                                 y_names"),e("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),e("span",{pre:!0,attrs:{class:"token string"}},[t._v('"salary"')]),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" valid_idx"),e("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),e("span",{pre:!0,attrs:{class:"token builtin"}},[t._v("list")]),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),e("span",{pre:!0,attrs:{class:"token builtin"}},[t._v("range")]),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),e("span",{pre:!0,attrs:{class:"token number"}},[t._v("800")]),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),e("span",{pre:!0,attrs:{class:"token number"}},[t._v("1000")]),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" bs"),e("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),e("span",{pre:!0,attrs:{class:"token number"}},[t._v("64")]),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\nlearn "),e("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" tabular_learner"),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("dls"),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n")])])]),e("h4",{staticClass:"doc_header",attrs:{id:"TabularLearner.predict"}},[e("code",[t._v("TabularLearner.predict")]),e("a",{staticClass:"source_link",staticStyle:{float:"right"},attrs:{href:"https://github.com/fastai/fastai/tree/master/fastai/tabular/learner.py#L14"}},[t._v("[source]")])]),t._v(" "),e("blockquote",[e("p",[e("code",[t._v("TabularLearner.predict")]),t._v("("),e("strong",[e("code",[t._v("row")])]),t._v(")")])]),t._v(" "),e("p",[t._v("Prediction on "),e("code",[t._v("item")]),t._v(", fully decoded, loss function decoded and probabilities")]),t._v(" "),e("p",[t._v("We can pass in an individual row of data into our "),e("RouterLink",{attrs:{to:"/tabular.learner.html#TabularLearner"}},[e("code",[t._v("TabularLearner")])]),t._v("'s "),e("code",[t._v("predict")]),t._v(" method. It's output is slightly different from the other "),e("code",[t._v("predict")]),t._v(" methods, as this one will always return the input as well:")],1),t._v(" "),e("div",{staticClass:"language-python extra-class"},[e("pre",{pre:!0,attrs:{class:"language-python"}},[e("code",[t._v("row"),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" clas"),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" probs "),e("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" learn"),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("predict"),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("df"),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("iloc"),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("[")]),e("span",{pre:!0,attrs:{class:"token number"}},[t._v("0")]),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("]")]),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n")])])]),e("div",{staticClass:"language-python extra-class"},[e("pre",{pre:!0,attrs:{class:"language-python"}},[e("code",[t._v("row"),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("show"),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n")])])]),e("table",{staticClass:"dataframe",attrs:{border:"1"}},[e("thead",[e("tr",{staticStyle:{"text-align":"right"}},[e("th"),t._v(" "),e("th",[t._v("workclass")]),t._v(" "),e("th",[t._v("education")]),t._v(" "),e("th",[t._v("marital-status")]),t._v(" "),e("th",[t._v("occupation")]),t._v(" "),e("th",[t._v("relationship")]),t._v(" "),e("th",[t._v("race")]),t._v(" "),e("th",[t._v("education-num_na")]),t._v(" "),e("th",[t._v("age")]),t._v(" "),e("th",[t._v("fnlwgt")]),t._v(" "),e("th",[t._v("education-num")]),t._v(" "),e("th",[t._v("salary")])])]),t._v(" "),e("tbody",[e("tr",[e("th",[t._v("0")]),t._v(" "),e("td",[t._v("Private")]),t._v(" "),e("td",[t._v("Assoc-acdm")]),t._v(" "),e("td",[t._v("Married-civ-spouse")]),t._v(" "),e("td",[t._v("#na#")]),t._v(" "),e("td",[t._v("Wife")]),t._v(" "),e("td",[t._v("White")]),t._v(" "),e("td",[t._v("False")]),t._v(" "),e("td",[t._v("49.0")]),t._v(" "),e("td",[t._v("101320.001685")]),t._v(" "),e("td",[t._v("12.0")]),t._v(" "),e("td",[t._v("<50k")])])])]),t._v(" "),e("div",{staticClass:"language-python extra-class"},[e("pre",{pre:!0,attrs:{class:"language-python"}},[e("code",[t._v("clas"),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" probs\n")])])]),e("div",{staticClass:"language- extra-class"},[e("pre",[e("code",[t._v("(tensor(0), tensor([0.5264, 0.4736]))\n")])])])])}),[],!1,null,null,null);a.default=r.exports}}]);