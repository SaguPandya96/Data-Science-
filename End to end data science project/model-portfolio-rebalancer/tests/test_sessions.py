from datetime import date

import pytest
from conftest import et

from rebalancer.sessions import ET, MarketCalendar, Session


@pytest.mark.parametrize(
    ("when", "session", "start", "end"),
    [
        (et(2026, 11, 3, 10, 0), Session.REGULAR, "Tue 09:30", "Tue 16:00"),
        (et(2026, 11, 3, 5, 0), Session.PRE, "Tue 04:00", "Tue 09:30"),
        (et(2026, 11, 3, 16, 0), Session.POST, "Tue 16:00", "Tue 20:00"),
        (et(2026, 11, 3, 23, 0), Session.OVERNIGHT, "Tue 20:00", "Wed 04:00"),
        (et(2026, 11, 4, 3, 59), Session.OVERNIGHT, "Tue 20:00", "Wed 04:00"),
        # Weekend runs Friday 20:00 to Sunday 20:00 before the exchange 23/5 day starts.
        (et(2026, 11, 6, 20, 0), Session.WEEKEND, "Fri 20:00", "Sun 20:00"),
        (et(2026, 11, 7, 12, 0), Session.WEEKEND, "Fri 20:00", "Sun 20:00"),
        (et(2026, 11, 8, 20, 0), Session.OVERNIGHT, "Sun 20:00", "Mon 04:00"),
        # Thanksgiving: closed, and so is the overnight session in front of it.
        (et(2026, 11, 25, 21, 0), Session.CLOSED, "Wed 20:00", "Thu 20:00"),
        (et(2026, 11, 26, 12, 0), Session.CLOSED, "Wed 20:00", "Thu 20:00"),
        (et(2026, 11, 26, 20, 30), Session.OVERNIGHT, "Thu 20:00", "Fri 04:00"),
        # Day after Thanksgiving closes at 13:00, after-hours at 17:00, then it's the weekend.
        (et(2026, 11, 27, 12, 59), Session.REGULAR, "Fri 09:30", "Fri 13:00"),
        (et(2026, 11, 27, 13, 0), Session.POST, "Fri 13:00", "Fri 17:00"),
        (et(2026, 11, 27, 17, 0), Session.WEEKEND, "Fri 17:00", "Sun 20:00"),
        # From Sunday Dec 6 the overnight opens at 21:00 and there's a daily 20:00-21:00 pause.
        (et(2026, 12, 6, 20, 30), Session.WEEKEND, "Fri 20:00", "Sun 21:00"),
        (et(2026, 12, 6, 21, 0), Session.OVERNIGHT, "Sun 21:00", "Mon 04:00"),
        (et(2026, 12, 8, 19, 59), Session.POST, "Tue 16:00", "Tue 20:00"),
        (et(2026, 12, 8, 20, 0), Session.PAUSE, "Tue 20:00", "Tue 21:00"),
        (et(2026, 12, 8, 21, 0), Session.OVERNIGHT, "Tue 21:00", "Wed 04:00"),
        # Friday evening after the cutover has no pause: it goes straight into the weekend.
        (et(2026, 12, 11, 20, 0), Session.WEEKEND, "Fri 20:00", "Sun 21:00"),
        # Good Friday 2027 and a Monday holiday fold into the weekend around them.
        (et(2027, 3, 26, 10, 0), Session.WEEKEND, "Thu 20:00", "Sun 21:00"),
        (et(2027, 1, 18, 10, 0), Session.WEEKEND, "Fri 20:00", "Mon 21:00"),
        (et(2027, 1, 18, 21, 0), Session.OVERNIGHT, "Mon 21:00", "Tue 04:00"),
        # Juneteenth 2026 is a Friday.
        (et(2026, 6, 19, 10, 0), Session.WEEKEND, "Thu 20:00", "Sun 20:00"),
    ],
)
def test_session_at(calendar, when, session, start, end):
    window = calendar.session_at(when)
    assert window.session is session
    assert window.start.astimezone(ET).strftime("%a %H:%M") == start
    assert window.end.astimezone(ET).strftime("%a %H:%M") == end


def test_daylight_saving_change_keeps_local_times(calendar):
    # DST ends Sunday Nov 1 2026; Monday's open is still 09:30 local, now 14:30 UTC.
    window = calendar.session_at(et(2026, 11, 2, 10, 0))
    assert window.start.hour == 14 and window.start.minute == 30


def test_hand_added_holiday_overrides_the_library():
    cal = MarketCalendar(extra_holidays={date(2026, 11, 4)})
    assert cal.session_at(et(2026, 11, 4, 10, 0)).session is Session.CLOSED
    assert cal.session_at(et(2026, 11, 3, 22, 0)).session is Session.CLOSED
    assert cal.session_at(et(2026, 11, 4, 20, 30)).session is Session.OVERNIGHT


def test_hand_added_half_day():
    cal = MarketCalendar(extra_half_days={date(2026, 11, 4)})
    assert cal.session_at(et(2026, 11, 4, 13, 30)).session is Session.POST


def test_last_regular_close_spans_the_weekend(calendar):
    close = calendar.last_regular_close(et(2026, 11, 9, 3, 0))
    assert close == et(2026, 11, 6, 16, 0)
    assert calendar.last_regular_close(et(2026, 11, 28, 12, 0)) == et(2026, 11, 27, 13, 0)


def test_naive_timestamps_are_refused(calendar):
    from datetime import datetime

    with pytest.raises(ValueError):
        calendar.session_at(datetime(2026, 11, 3, 10, 0))
