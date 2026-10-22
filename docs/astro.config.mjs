// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// Published on GitHub Pages at https://misterekkk.github.io/lignum/ (see .github/workflows/docs.yml).
export default defineConfig({
  site: 'https://misterekkk.github.io',
  base: '/lignum',
  integrations: [
    starlight({
      title: 'lignum',
      description: 'Bit-exact, fast inference for LightGBM and XGBoost models.',
      logo: { src: './assets/logo.svg' },
      favicon: '/favicon.svg',
      social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/misterekkk/lignum' }],
      editLink: { baseUrl: 'https://github.com/misterekkk/lignum/edit/dev/v0.2/docs/' },
      customCss: ['./src/styles/custom.css'],
      sidebar: [
        { label: 'Start', items: ['guides/getting-started'] },
        {
          label: 'Guides',
          items: ['guides/models', 'guides/predicting', 'guides/arrow', 'guides/compiled', 'guides/c-api'],
        },
        { label: 'Concepts', items: ['concepts/exactness', 'concepts/internals'] },
        { label: 'Benchmarks', items: ['benchmarks'] },
        { label: 'Reference', items: ['reference/python'] },
      ],
    }),
  ],
});
