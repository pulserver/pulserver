# Development workflow

1. Fork `pulserver/pulserver` on GitHub and clone the fork.
2. Add `https://github.com/pulserver/pulserver.git` as the `upstream` remote.
3. Update local `main`, then create a focused feature branch.
4. Commit tested changes and push the branch to the fork.
5. Open a pull request against `pulserver/pulserver:main`.

Keep commits reviewable and avoid unrelated formatting or prose churn. A change
that belongs to sequence design is contributed to
[pypulseqpp](https://github.com/pulserver/pypulseqpp), and one that belongs to
a reconstruction algorithm to [bartorch](https://github.com/mcencini/bartorch).
