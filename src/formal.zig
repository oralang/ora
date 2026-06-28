//! Module root for formal compiler models and snapshot emitters.
//!
//! Files under `src/formal/` stay standalone and do not reach upward with
//! `../sema/...` imports, because Zig treats that as outside their module path.
//! This module is rooted at `src/` and exposes the narrow compiler facts those
//! formal generators and manifest models need.

const model = @import("sema/model.zig");
const ora_types = @import("ora_types");
const refinements = @import("ora_refinements");
const region = @import("sema/region.zig");
const type_descriptors = @import("sema/type_descriptors.zig");

pub const obligation = @import("formal/obligation.zig");
pub const obligation_crosscheck = @import("formal/obligation_crosscheck.zig");
pub const obligation_dump = @import("formal/obligation_dump.zig");
pub const obligation_to_lean = @import("formal/obligation_to_lean.zig");

pub const builtin = ora_types.builtin;
pub const region_assign = ora_types.region_assign;
pub const semantic = ora_types.semantic;
pub const refinement_registry = refinements;

pub const LocatedType = model.LocatedType;
pub const RefinementArg = model.RefinementArg;
pub const Type = model.Type;

pub const typeEql = type_descriptors.typeEql;
pub const typesAssignable = type_descriptors.typesAssignable;
pub const locatedTypeEql = region.locatedTypeEql;
pub const locatedTypeAssignable = region.isAssignable;
