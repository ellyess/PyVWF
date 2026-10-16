Review this proposal for PyVWF.

The repository is checked out at /var/folders/hb/8jmqlr8502jd4_jj817kmrq00000gn/T/pyvwf-checkouts/a75ab221. Treat that directory as the
repository root: resolve every path in your instructions and in the proposal
against it, and read nothing outside it.

After your review, end your reply with a line "Files read:" followed by the
absolute path of every file you opened, one per line.

## Proposal

**Claim:** Every country-level figure can name the grid it was computed on,
once an audit records which grid file each region uses and how many points it
holds.

**Mechanism:** Two figures for one region can differ mostly because of the
grid: a uniform lattice averages over points that carry no fleet, a
fleet-weighted grid drops them. A figure without its grid cannot be compared.

**Change:** A script, `scripts/analysis/grid_audit.py`, writes one row per
country-level region (FR, BE, IE, SE, NO, ES, IT, PT, NL): the grid file in
use, its point count, and its total capacity. For each region it lists
`PyVWFPaths.COUNTRY_LEVEL_DATA / "grid_points" / <code>` for files matching
`<code>_grid_points_20*.csv`, sorts the matches, and takes the last as the
current grid. The audit's table is then cited beside every country-level
figure, and two previously reported Belgian figures are labelled with their
grids from it.

**Test:** No training or scoring: the audit reads input files only. Checked
by hand against two regions.

**Gates:** None; this is a record, not a result.

**Expected effect and size:** One table of nine rows. It should show which
regions are on a fleet-weighted grid and which on the uniform one.

**Risks:** A region without year-specific files would be missed; it is
reported as missing.
