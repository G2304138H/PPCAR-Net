# Publishing the project page

The website deploys through `.github/workflows/pages.yml`. It packages only tracked
`project-page/` and `figures/` files. Checkpoints, datasets, training code and local
planning records are not included in the website artifact.

## First deployment

1. Commit and push the workflow and final website changes to `main`.
2. When ready to release, make the repository public.
3. In GitHub, open **Settings → Pages → Build and deployment → Source** and select **GitHub Actions**.
4. Open **Actions → Deploy project page → Run workflow**, select `main`, and run it.
5. Wait for the deployment to succeed. Open the URL shown by the deployment.

Website: https://G2304138H.github.io/vessel-code/

The root redirects to `project-page/`. The full gallery remains available at
`figures/video_demo/index.html`; the existing relative image, video and return
links continue to work. Later pushes changing the website, figures or workflow
on `main` trigger another deployment automatically.

Verify the opening animation, RCA/LCA case switching, method animations, diseased
CT example, real X-ray videos, complete comparison gallery, and return link in a
signed-out browser. The Paper button remains a placeholder until an arXiv URL is added.

The workflow requires Pages to be enabled first. If it ran before configuration,
finish step 3 and rerun it. GitHub Free supports Pages for public repositories;
private repository support depends on your plan.
