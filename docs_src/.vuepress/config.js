module.exports = {
  title: '',
  description: 'Welcome to fastai', 
  themeConfig: {
    logo: 'https://images.exxactcorp.com/CMS/landing-page/resource-center/supported-software/logo/Deep-Learning/fastai-logo.png',
    repo: 'fastai/fastai',
    repoLabel: 'GitHub',
    sidebar: 'auto',
    sidebarDepth: 4,
    displayAllHeaders: true, 
    nav: [
      { text: 'Getting Started', link: '/' },
      { text: 'Tutorials', link: '/tutorials/' },
      { text: 'Training', link: '/training/' },
      {
        text: 'Data',
        items: [
          {text: 'Data Blocks', link: '/data/06_data.block'},
          {text: 'Data Transforms', link: '/data/05_data.transforms'},
          {text: 'Data External', link: '/data/04_data.external'},
          {text: 'Data Core', link: '/data/03_data.core'},
          {text: 'DataLoaders', link: '/data/06_data.load'},
        ] },
      { text: 'Core', link: '/core/' },
      { text: 'Vision', link: '/vision/' },
      { text: 'Text', link: '/text/' },
      { text: 'Tabular', link: '/tabular/' },
      { text: 'Medical', link: '/medical/' },
    ]
  }
}
