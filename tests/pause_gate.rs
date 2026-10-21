use auv::pause::PauseGate;

#[test]
fn toggle_reset_and_interrupt() {
    let gate = PauseGate::new();
    assert!(!gate.is_paused());
    assert!(gate.toggle());
    assert!(gate.is_paused());

    // 暂停中 interrupted 立即返回 true，不必等到恢复
    assert!(gate.wait_while_paused(|| true));

    gate.reset();
    assert!(!gate.is_paused());
    assert!(!gate.wait_while_paused(|| true));
}
