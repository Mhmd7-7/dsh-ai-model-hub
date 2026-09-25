/**
 * The hub's shared runtime vocabulary.
 *
 * These types describe the *host* side of a model — is it configured, is it
 * running, is it healthy, may the router pick it — as opposed to the *descriptor*
 * side, which is static configuration.
 *
 * @module dsh-ai-model-hub/types
 */
/**
 * Whether a model can serve work right now.
 *
 * Kept distinct from {@link LifecycleState} because they answer different
 * questions: `availability` is the routing input ("may I choose this?"), while
 * `lifecycle` is the management input ("is there a process, and do I own it?").
 * A model with an external endpoint the hub never started is `available` with
 * `lifecycle: 'external'`.
 */
export const AVAILABILITY_STATES = [
    /** Ready to serve. */
    'available',
    /** Known and configured, but its process is not running. Startable models live here. */
    'stopped',
    /** A start or health probe is in flight. */
    'starting',
    /** Running but its health check fails. */
    'unhealthy',
    /** Disabled in configuration; never selected. */
    'disabled',
    /** This machine cannot satisfy the model's declared resource needs. */
    'unsupported',
    /** The descriptor is broken or its adapter is missing. */
    'error',
];
/** Which process, if any, is behind a model. */
export const LIFECYCLE_STATES = [
    /** No process; nothing known about liveness. */
    'not_running',
    /** A process owned by the hub is starting. */
    'starting',
    /** A process owned by the hub is running. */
    'running',
    /** A process is stopping. */
    'stopping',
    /** A process failed and the hub gave up on it. */
    'failed',
    /** Reached over the network; the hub did not start it and does not manage it. */
    'external',
];
/**
 * Subtract what is already resident from what the machine has.
 *
 * Only the *available* pair is reduced. Capacity is a property of the hardware
 * and does not change because a model is loaded — rewriting it would be the
 * "claiming more than the machine offers" failure the probe is written to avoid,
 * in reverse.
 *
 * An undefined available value stays undefined: "we never measured it" is not
 * the same fact as "there is none left", and the caller must be able to tell.
 *
 * @param profile - the probed profile.
 * @param use - what the resident models hold.
 * @returns a profile whose available figures exclude `use`.
 */
export function reserveResources(profile, use) {
    const reserved = {};
    if (profile.availableVramGb !== undefined) {
        reserved.availableVramGb = Math.max(0, round1(profile.availableVramGb - (use.vramGb ?? 0)));
    }
    if (profile.availableRamGb !== undefined) {
        reserved.availableRamGb = Math.max(0, round1(profile.availableRamGb - (use.ramGb ?? 0)));
    }
    return { ...profile, ...reserved };
}
/**
 * Round to one decimal place, the precision every resource figure is reported at.
 * @param value - the number to round.
 * @returns the rounded value.
 */
function round1(value) {
    return Math.round(value * 10) / 10;
}
