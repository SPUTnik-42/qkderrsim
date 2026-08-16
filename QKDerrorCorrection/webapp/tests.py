from django.test import TestCase
from django.urls import reverse


class MultihopSimulatorViewTests(TestCase):
    def test_multihop_simulator_page_renders(self):
        response = self.client.get(reverse('multihop_simulator'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Multi-Hop')

    def test_multihop_simulator_accepts_post_data(self):
        payload = {
            'protocol': ['bb84', 'dps'],
            'pa_protocol': 'toeplitz',
            'num_qubits': '2000',
            'freq': '1e7',
            'mean_photon': '0.1',
            'att_db_km': '0.2',
            'det_eff': '0.8',
            'dark_count': '0.01',
            'chain_data': '[{"distance": 50, "capacity": 1000000000}, {"distance": 50, "capacity": 1000000000}]',
            'total_distance': '100',
            'relay_counts': '1',
            'sifting_exchanges': '3',
            'overhead_factor': '3',
            'packet_size': '10000',
        }
        response = self.client.post(reverse('multihop_simulator'), payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn('effective_rate', response.context)
        self.assertEqual(response.context['total_distance'], 100.0)
        self.assertEqual(response.context['form_data']['protocol'], 'bb84')

    def test_multihop_simulator_uses_chain_distance_from_gui(self):
        payload = {
            'protocol': 'bb84',
            'pa_protocol': 'toeplitz',
            'num_qubits': '2000',
            'freq': '1e7',
            'mean_photon': '0.1',
            'att_db_km': '0.2',
            'det_eff': '0.8',
            'dark_count': '0.01',
            'chain_data': '[{"distance": 30, "capacity": 1000000000}, {"distance": 20, "capacity": 1000000000}, {"distance": 40, "capacity": 1000000000}]',
            'total_distance': '90',
            'relay_counts': '2',
            'sifting_exchanges': '3',
            'overhead_factor': '3',
            'packet_size': '10000',
        }
        response = self.client.post(reverse('multihop_simulator'), payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['total_distance'], 90.0)
