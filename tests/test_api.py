import os
import tempfile
import unittest


class CorpdeskIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        os.environ["APP_DB_PATH"] = os.path.join(cls.tmpdir.name, "test.db")
        from app import create_app

        cls.app = create_app()
        cls.client = cls.app.test_client()

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def login(self, employee_id="user1", role="user"):
        return self.client.post(
            "/auth/login",
            json={"employee_id": employee_id, "name": employee_id, "role": role},
        )

    def test_login_and_me(self):
        res = self.login("tester1", "manager")
        self.assertEqual(res.status_code, 200)
        me = self.client.get("/auth/me")
        self.assertEqual(me.status_code, 200)
        body = me.get_json()
        self.assertTrue(body["logged_in"])
        self.assertEqual(body["user"]["employee_id"], "tester1")

    def test_approval_from_chatbot_flow(self):
        self.login("drafter1", "user")
        created = self.client.post(
            "/approval/from-chatbot",
            json={
                "query": "재고 보고서 작성해줘",
                "answer": "재고는 정상입니다.",
                "label": "재고",
                "approvers": ["manager1"],
            },
        )
        self.assertEqual(created.status_code, 200)
        doc_id = created.get_json()["document_id"]
        submit = self.client.post(f"/approval/documents/{doc_id}/submit")
        self.assertEqual(submit.status_code, 200)

        self.login("drafter1", "user")
        forbidden = self.client.post(
            f"/approval/documents/{doc_id}/decision",
            json={"decision": "approved", "comment": "권한없음"},
        )
        self.assertEqual(forbidden.status_code, 403)

        self.login("manager1", "manager")
        decision = self.client.post(
            f"/approval/documents/{doc_id}/decision",
            json={"decision": "approved", "comment": "확인"},
        )
        self.assertEqual(decision.status_code, 200)

        doc = self.client.get(f"/approval/documents/{doc_id}")
        self.assertEqual(doc.status_code, 200)
        self.assertEqual(doc.get_json()["document"]["status"], "approved")

if __name__ == "__main__":
    unittest.main()

