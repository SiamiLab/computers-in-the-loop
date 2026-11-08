# Permanent links (w3id.org)

The QR codes in the book point to `https://w3id.org/citl/chNN`. w3id.org is a free,
community-run permanent-address service: it redirects to wherever the material lives now, so
printed QR codes keep working if the hosting changes.

## Register (once)

1. Fork https://github.com/perma-id/w3id.org on GitHub.
2. Copy the `citl/` folder (with `.htaccess` and `README.md`) into the root of your fork.
3. Open a pull request titled "Add citl (Computers in the Loop)". It is usually merged within a
   few days.
4. Test: https://w3id.org/citl/ch01 should open the Chapter 1 page.

The QR codes are already printed in the book; they start working once the pull request is merged.

If the site moves, edit the target addresses in `.htaccess` (another small pull request); the
QR codes in the book do not change.
